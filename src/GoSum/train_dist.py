from datautils import *
from model import *
from model2 import SGraph
from utils import *
import os
import pickle
from tqdm import tqdm
import time

import os, sys

from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from rouge_score import rouge_scorer

# import warnings
# warnings.filterwarnings("ignore")
import copy

import argparse

def update_moving_average( m_ema, m, decay ):
    with torch.no_grad():
        param_dict_m_ema =  m_ema.module.parameters()  if isinstance(  m_ema, DDP ) else m_ema.parameters() 
        param_dict_m =  m.module.parameters()  if isinstance( m , DDP ) else  m.parameters() 
        for param_m_ema, param_m in zip( param_dict_m_ema, param_dict_m ):
            param_m_ema.copy_( decay * param_m_ema + (1-decay) *  param_m )

def LOG( info, end="\n" ):
    global log_out_file
    with open( log_out_file, "a" ) as f:
        f.write( info + end )

def load_corpus( fname, is_training  ):
    corpus = []
    with open( fname, "r" ) as f:
        for line in tqdm(f):
            data = json.loads(line)
            if len(data["text"]) == 0 or len(data["summary"]) == 0:
                continue
            if is_training:
                if len( data["indices"] ) == 0 or len( data["score"] ) == 0:
                    continue

            corpus.append( data )
    return corpus


############################################
##  Loading args_
############################################
# num_gpus = torch.cuda.device_count()
# print(num_gpus)
parser = argparse.ArgumentParser()
parser.add_argument("-config_file_path",  default= 'config/pubmeni/training.config')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

args = parser.parse_args()

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device=torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

args_ = Dict2Class(json.load(open(args.config_file_path)))

training_corpus_file_name  = args_.training_corpus_file_name
validation_corpus_file_name = args_.validation_corpus_file_name
model_folder = args_.model_folder
log_folder = args_.log_folder
vocabulary_file_name = args_.vocabulary_file_name
pretrained_unigram_embeddings_file_name = args_.pretrained_unigram_embeddings_file_name
num_heads = args_.num_heads
hidden_dim = args_.hidden_dim
N_enc_l = args_.N_enc_l
N_enc_g = args_.N_enc_g
N_dec = args_.N_dec
max_seq_len = args_.max_seq_len
max_doc_len = args_.max_doc_len
num_of_epochs = args_.num_of_epochs
print_every = args_.print_every
save_every = args_.save_every
validate_every = args_.validate_every
restore_old_checkpoint = args_.restore_old_checkpoint
learning_rate = args_.learning_rate
warmup_step = args_.warmup_step
weight_decay = args_.weight_decay
dropout_rate = args_.dropout_rate
n_device = args_.n_device
batch_size_per_device = args_.batch_size_per_device
max_extracted_sentences_per_document = args_.max_extracted_sentences_per_document
moving_average_decay = args_.moving_average_decay
p_stop_thres = args_.p_stop_thres


if not os.path.exists( log_folder ):
    os.makedirs(log_folder)
if not os.path.exists( model_folder ):
    os.makedirs(model_folder)
log_out_file = log_folder + "/train.log"

training_corpus = load_corpus( training_corpus_file_name, True )
validation_corpus = load_corpus( validation_corpus_file_name, False )

with open( vocabulary_file_name, "rb") as f:
    words = pickle.load(f)
with open(pretrained_unigram_embeddings_file_name, "rb") as f:
    pretrained_embedding = pickle.load(f)
vocab = Vocab(words)
vocab_size, embed_dim = pretrained_embedding.shape

############################################
##  Setting Dataset
############################################
train_dataset = ExtractionTrainingDataset(  training_corpus,  vocab , max_seq_len,  max_doc_len)
train_sampler = DistributedSampler(train_dataset)
train_data_loader = DataLoader( 
    train_dataset, sampler=train_sampler,
    batch_size=batch_size_per_device, num_workers=4,  
    drop_last= True, worker_init_fn = lambda x:[np.random.seed( int( time.time() )+x ), 
    torch.manual_seed(int( time.time() ) + x) ],  pin_memory = True, collate_fn=collate_fn)

total_number_of_samples = train_dataset.__len__()
val_dataset = ExtractionValidationDataset( validation_corpus, vocab, max_seq_len, max_doc_len )
val_sampler = DistributedSampler(val_dataset)
val_data_loader = DataLoader( 
    val_dataset, sampler=val_sampler,
    batch_size=batch_size_per_device, num_workers=4,  
    drop_last= False,  worker_init_fn = lambda x:[np.random.seed( int( time.time() ) + 1 + x ), 
    torch.manual_seed( int( time.time() ) + 1 + x ) ],  pin_memory = True, collate_fn=collate_fn_valid)


############################################
##  Define Models
############################################
local_sentence_encoder = LocalSentenceEncoder( vocab_size, vocab.pad_index, embed_dim, num_heads, hidden_dim, N_enc_l, pretrained_embedding )
local_section_encoder = LocalSentenceEncoder( vocab_size, vocab.pad_index, embed_dim, num_heads, hidden_dim, N_enc_l, pretrained_embedding )
global_context_encoder = GlobalContextEncoder( embed_dim, num_heads, hidden_dim, N_enc_g )
graph_encoder = SGraph( embed_dim )

extraction_context_decoder = ExtractionContextDecoder( embed_dim, num_heads, hidden_dim, N_dec )
extractor = Extractor( embed_dim, num_heads )

# restore most recent checkpoint
if restore_old_checkpoint:
    ckpt = load_model( model_folder )
else:
    ckpt = None

if ckpt is not None:
    local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
    local_section_encoder.load_state_dict( ckpt["local_section_encoder"] )
    graph_encoder.load_state_dict( ckpt["graph_encoder"] )
    global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
    extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
    extractor.load_state_dict( ckpt["extractor"] )
    LOG("model restored!")
    print("model restored!")

# device = torch.device(  "cuda:%d"%( gpu_list[0] ) if torch.cuda.is_available() else "cpu" )
device = torch.device("cuda", args.local_rank)

local_sentence_encoder_ema = copy.deepcopy( local_sentence_encoder ).to(device)
local_section_encoder_ema = copy.deepcopy( local_section_encoder ).to(device)
graph_encoder_ema = copy.deepcopy( graph_encoder ).to(device)
global_context_encoder_ema = copy.deepcopy( global_context_encoder ).to(device)
extraction_context_decoder_ema = copy.deepcopy( extraction_context_decoder ).to(device)
extractor_ema = copy.deepcopy( extractor ).to(device)

local_sentence_encoder.to(device)
local_section_encoder.to(device)
graph_encoder.to(device)
global_context_encoder.to(device)
extraction_context_decoder.to(device)
extractor.to(device)

if device.type == "cuda" and n_device > 1:
    local_sentence_encoder = DDP( local_sentence_encoder, device_ids=[args.local_rank], find_unused_parameters=True)
    local_section_encoder = DDP( local_section_encoder, device_ids=[args.local_rank], find_unused_parameters=True )
    graph_encoder = DDP(graph_encoder, device_ids=[args.local_rank], find_unused_parameters=True)
    global_context_encoder = DDP( global_context_encoder, device_ids=[args.local_rank], find_unused_parameters=True )
    extraction_context_decoder = DDP( extraction_context_decoder, device_ids=[args.local_rank], find_unused_parameters=True )
    extractor = DDP( extractor, device_ids=[args.local_rank], find_unused_parameters=True )

    local_sentence_encoder_ema = DDP( local_sentence_encoder_ema, device_ids=[args.local_rank], find_unused_parameters=True )
    local_section_encoder_ema = DDP( local_section_encoder_ema, device_ids=[args.local_rank], find_unused_parameters=True )
    graph_encoder_ema = DDP( graph_encoder_ema, device_ids=[args.local_rank], find_unused_parameters=True )
    global_context_encoder_ema = DDP( global_context_encoder_ema, device_ids=[args.local_rank], find_unused_parameters=True )
    extraction_context_decoder_ema = DDP( extraction_context_decoder_ema, device_ids=[args.local_rank], find_unused_parameters=True )
    extractor_ema = DDP( extractor_ema, device_ids=[args.local_rank], find_unused_parameters=True )    

    model_parameters = [ par for par in local_sentence_encoder.module.parameters() if par.requires_grad  ]  + \
                    [ par for par in local_section_encoder.module.parameters() if par.requires_grad  ]  + \
                    [ par for par in graph_encoder.module.parameters() if par.requires_grad  ]  + \
                    [ par for par in global_context_encoder.module.parameters() if par.requires_grad  ]   + \
                    [ par for par in extraction_context_decoder.module.parameters() if par.requires_grad  ]  + \
                    [ par for par in extractor.module.parameters() if par.requires_grad  ]  
else:
    model_parameters =  [ par for par in local_sentence_encoder.parameters() if par.requires_grad  ]  + \
                    [ par for par in local_section_encoder.parameters() if par.requires_grad  ]  + \
                    [ par for par in graph_encoder.parameters() if par.requires_grad  ]  + \
                    [ par for par in global_context_encoder.parameters() if par.requires_grad  ]   + \
                    [ par for par in extraction_context_decoder.parameters() if par.requires_grad  ]  + \
                    [ par for par in extractor.parameters() if par.requires_grad  ]  

optimizer = torch.optim.Adam( model_parameters , lr= learning_rate , weight_decay = weight_decay )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  min( (x+1)**(-0.5), (x+1)*(warmup_step**(-1.5))  ), last_epoch=-1, verbose=True)

if ckpt is not None:
    try:
        optimizer.load_state_dict( ckpt["optimizer"] )
        scheduler.load_state_dict( ckpt["scheduler"] )
        LOG("optimizer restored!")
        print("optimizer restored!")
    except:
        pass

current_epoch = 0
current_batch = 0

if ckpt is not None:
    current_batch = ckpt["current_batch"]
    current_epoch = int( current_batch * batch_size_per_device * n_device / total_number_of_samples )
    LOG("current_batch restored!")
    print("current_batch restored!")

np.random.seed()

rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)


def train_iteration(batch):
    seqs, doc_mask, selected_y_label, selected_score, summary_seq, valid_sen_idxs, seqs_sec, sec_mask, sbelong, G, nsents, nsecs = batch
    # seqs, doc_mask, selected_y_label, selected_score, summary_seq, valid_sen_idxs = batch
    # seqs / summary_seq: bs x 500 x 100 torch tensor  
    # doc_mask / selected_y_label / valid_sen_idxs: bs x 500 torch tensor
    # selected_score: bs  torch tensor
    # print(sum(nsents))
    bsize = seqs.size(0)
    seqs = seqs.to(device)
    doc_mask = doc_mask.to(device)
    seqs_sec = seqs_sec.to(device)
    sec_mask = sec_mask.to(device)
    # ipdb.set_trace()
    G = G.to(device)
    # snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
    # print(len(snode_id) + 10000)

    selected_y_label = selected_y_label.to(device)
    selected_score = selected_score.to(device)
    
    valid_sen_idxs_np = valid_sen_idxs.detach().cpu().numpy()
    valid_sen_idxs = -1*np.ones_like( valid_sen_idxs_np )
    valid_sen_idxs_np[ valid_sen_idxs_np>=doc_mask.size(1) ] = -1
    for doc_i in range( valid_sen_idxs_np.shape[0] ):
        valid_idxs = valid_sen_idxs_np[doc_i][ valid_sen_idxs_np[doc_i] != -1]
        valid_sen_idxs[doc_i, : len(valid_idxs)] = valid_idxs
    valid_sen_idxs = torch.from_numpy(valid_sen_idxs).to(device)
    
    num_documents = seqs.size(0)
    num_sentences = seqs.size(1)
    num_sections = seqs_sec.size(1)
    
    local_sen_embed = local_sentence_encoder( seqs.view(-1, seqs.size(2) ) , dropout_rate )
    local_sen_embed = local_sen_embed.view( -1, num_sentences, local_sen_embed.size(1) )
    local_sec_embed = local_section_encoder( seqs_sec.view(-1, seqs_sec.size(2) ) , dropout_rate )
    local_sec_embed = local_sec_embed.view( -1, num_sections, local_sec_embed.size(1) )



    # GNN module
    sent_state = torch.zeros((sum(nsents), local_sen_embed.shape[2])).to(device)
    sec_state = torch.zeros((sum(nsecs), local_sec_embed.shape[2])).to(device)

    len_count = 0
    for i, slen in enumerate(nsents):
        sent_state[len_count: len_count + slen] = local_sen_embed[i][:slen]
        len_count += slen

    len_count = 0
    for i, clen in enumerate(nsecs):
        sec_state[len_count: len_count + clen] = local_sec_embed[i][:clen]
        len_count += clen

    global_sen_embed_, global_sec_embed_, global_gsen_embed_ = graph_encoder(G, sent_state, sec_state, sent_state)

    global_sen_embed = torch.zeros_like(local_sen_embed)
    global_sec_embed = torch.zeros_like(local_sec_embed)
    global_gsen_embed = torch.zeros_like(local_sen_embed)

    len_count = 0
    for i, slen in enumerate(nsents):
        global_sen_embed[i][:slen] = global_sen_embed_[len_count: len_count + slen]
        global_gsen_embed[i][:slen] = global_gsen_embed_[len_count: len_count + slen]
        len_count += slen

    len_count = 0
    for i, clen in enumerate(nsecs):
        global_sec_embed[i][:clen] = global_sec_embed_[len_count: len_count + clen]
        len_count += clen

    # ipdb.set_trace()


    sec_embed_per_sent = torch.zeros_like(local_sen_embed)
    for i in range(bsize):
        sec_embed_per_sent[i][:len(sbelong[i])] = global_sec_embed[i][sbelong[i]]

    global_context_embed = global_context_encoder( global_gsen_embed, doc_mask , dropout_rate )

    
    doc_mask_np = doc_mask.detach().cpu().numpy()
    remaining_mask_np = np.ones_like( doc_mask_np ).astype( np.bool ) | doc_mask_np
    extraction_mask_np = np.zeros_like( doc_mask_np ).astype( np.bool ) | doc_mask_np
    
    log_action_prob_list = []
    log_stop_prob_list = []
    
    done_list = []
    extraction_context_embed = None
    
    for step in range(valid_sen_idxs.shape[1]):
        remaining_mask = torch.from_numpy( remaining_mask_np ).to(device)
        extraction_mask = torch.from_numpy( extraction_mask_np ).to(device)
        if step > 0:
            extraction_context_embed = extraction_context_decoder( global_sen_embed, remaining_mask, extraction_mask, dropout_rate )
        # ipdb.set_trace()
        p, p_stop, baseline = extractor( global_sen_embed, sec_embed_per_sent, global_context_embed, extraction_context_embed , extraction_mask , dropout_rate )
        # p(bs x 500) / p_stop(bs x 1)
        p_stop = p_stop.unsqueeze(1)
        m_stop = Categorical( torch.cat( [ 1-p_stop, p_stop  ], dim =1 ) )
        
        sen_indices = valid_sen_idxs[:, step]
        done = sen_indices == -1
        if len(done_list) > 0:
            done = torch.logical_or(done_list[-1], done)
            just_stop = torch.logical_and( ~done_list[-1], done )
        else:
            just_stop = done
        
        if torch.all( done ) and not torch.any(just_stop):
            break
        
        p = p.masked_fill( extraction_mask, 1e-12 )  
        normalized_p = p / p.sum(dim=1, keepdims = True)
        ## Here the sen_indices is actually pre-sampled action
        normalized_p = normalized_p[ np.arange( num_documents ), sen_indices ]
        log_action_prob = normalized_p.masked_fill( done, 1.0 ).log()
        
        log_stop_prob = m_stop.log_prob( done.to(torch.long)  )
        log_stop_prob = log_stop_prob.masked_fill( torch.logical_xor( done, just_stop ), 0.0 )
        
        log_action_prob_list.append( log_action_prob.unsqueeze(1) )
        log_stop_prob_list.append( log_stop_prob.unsqueeze(1) )
        done_list.append(done)
        
        for doc_i in range( num_documents ):
            sen_i = sen_indices[ doc_i ].item()
            if sen_i != -1:
                remaining_mask_np[doc_i,sen_i] = False
                extraction_mask_np[doc_i,sen_i] = True
    
        
    log_action_prob_list = torch.cat( log_action_prob_list, dim = 1 )
    log_stop_prob_list = torch.cat( log_stop_prob_list, dim = 1 )
    log_prob_list = log_action_prob_list + log_stop_prob_list


    if args_.apply_length_normalization:
        log_prob_list = log_prob_list.sum(dim=1)  / ( (log_prob_list != 0).to(torch.float32).sum(dim=1) )  
    else:
        log_prob_list = log_prob_list.sum(dim=1) 

    loss = (-log_prob_list * selected_score).mean()
    
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()

    return loss.item()


def validation_iteration(batch):
    seqs, doc_mask, summary_seq, seqs_sec, sec_mask, sbelong, G, nsents, nsecs = batch
    seqs = seqs.to(device)
    doc_mask = doc_mask.to(device)
    seqs_sec = seqs_sec.to(device)
    sec_mask = sec_mask.to(device)
    G = G.to(device)

    num_sentences = seqs.size(1)
    bsize = seqs.size(0)
    num_sections = seqs_sec.size(1)

    local_sen_embed  = local_sentence_encoder_ema( seqs.view(-1, seqs.size(2) ) )
    local_sen_embed = local_sen_embed.view( -1, num_sentences, local_sen_embed.size(1) )
    local_sec_embed = local_section_encoder_ema( seqs_sec.view(-1, seqs_sec.size(2) ) )
    local_sec_embed = local_sec_embed.view( -1, num_sections, local_sec_embed.size(1) )


    sent_state = torch.zeros((sum(nsents), local_sen_embed.shape[2])).to(device)
    sec_state = torch.zeros((sum(nsecs), local_sec_embed.shape[2])).to(device)

    len_count = 0
    for i, slen in enumerate(nsents):
        sent_state[len_count: len_count + slen] = local_sen_embed[i][:slen]
        len_count += slen

    len_count = 0
    for i, clen in enumerate(nsecs):
        sec_state[len_count: len_count + clen] = local_sec_embed[i][:clen]
        len_count += clen

    global_sen_embed_, global_sec_embed_, global_gsen_embed_ = graph_encoder_ema(G, sent_state, sec_state, sent_state)

    global_sen_embed = torch.zeros_like(local_sen_embed)
    global_sec_embed = torch.zeros_like(local_sec_embed)
    global_gsen_embed = torch.zeros_like(local_sen_embed)

    len_count = 0
    for i, slen in enumerate(nsents):
        global_sen_embed[i][:slen] = global_sen_embed_[len_count: len_count + slen]
        global_gsen_embed[i][:slen] = global_gsen_embed_[len_count: len_count + slen]
        len_count += slen

    len_count = 0
    for i, clen in enumerate(nsecs):
        global_sec_embed[i][:clen] = global_sec_embed_[len_count: len_count + clen]
        len_count += clen



    sec_embed_per_sent = torch.zeros_like(local_sen_embed)
    for i in range(bsize):
        sec_embed_per_sent[i][:len(sbelong[i])] = global_sec_embed[i][sbelong[i]]
    
    global_context_embed = global_context_encoder_ema( global_gsen_embed, doc_mask  )
    
    num_documents = seqs.size(0)
    doc_mask = doc_mask.detach().cpu().numpy()
    remaining_mask_np = np.ones_like( doc_mask ).astype( np.bool ) | doc_mask
    extraction_mask_np = np.zeros_like( doc_mask ).astype( np.bool ) | doc_mask
    
    seqs = seqs.detach().cpu().numpy()
    summary_seq = summary_seq.detach().cpu().numpy()
    
    done_list = []
    extraction_context_embed = None
    
    for step in range(max_extracted_sentences_per_document):
        remaining_mask = torch.from_numpy( remaining_mask_np ).to(device)
        extraction_mask = torch.from_numpy( extraction_mask_np ).to(device)
        if step > 0:
            extraction_context_embed = extraction_context_decoder_ema( global_sen_embed, remaining_mask, extraction_mask )
        p, p_stop, baseline = extractor_ema( global_sen_embed, sec_embed_per_sent, global_context_embed, extraction_context_embed , extraction_mask  )
        
        p = p.masked_fill( extraction_mask, 1e-12 )  
        normalized_p = p / (p.sum(dim=1, keepdims = True))

        stop_action = p_stop > p_stop_thres
        
        done = stop_action | torch.all(extraction_mask, dim = 1) 
        if len(done_list) > 0:
            done = torch.logical_or(done_list[-1], done)
        if torch.all( done ):
            break
            
        sen_indices = torch.argmax(normalized_p, dim =1)
        done_list.append(done)
        
        for doc_i in range( num_documents ):
            if not done[doc_i]:
                sen_i = sen_indices[ doc_i ].item()
                remaining_mask_np[doc_i,sen_i] = False
                extraction_mask_np[doc_i,sen_i] = True
                
    scores = []
    for doc_i in range(seqs.shape[0]):
        ref = "\n".join( [ vocab.seq2sent( seq ) for seq in summary_seq[doc_i] ]  ).strip()
        extracted_sen_indices = np.argwhere( remaining_mask_np[doc_i] == False )[:,0]
        hyp = "\n".join(  [ vocab.seq2sent( seq ) for seq in seqs[doc_i][extracted_sen_indices]] ).strip()
    
        score = rouge_cal.score( hyp, ref )
        scores.append( (score["rouge1"].fmeasure,score["rouge2"].fmeasure,score["rougeLsum"].fmeasure) )

    return scores


for epoch in range( current_epoch, num_of_epochs ):
    running_loss = 0 
    
    for count, batch in enumerate(train_data_loader):
        loss = train_iteration(batch)
        running_loss += loss

        # scheduler.step()

        update_moving_average(  local_sentence_encoder_ema,  local_sentence_encoder, moving_average_decay)
        update_moving_average(  local_section_encoder_ema,  local_section_encoder, moving_average_decay)
        update_moving_average(  graph_encoder_ema,  graph_encoder, moving_average_decay)
        update_moving_average(  global_context_encoder_ema ,  global_context_encoder, moving_average_decay)
        update_moving_average(  extraction_context_decoder_ema,  extraction_context_decoder, moving_average_decay)
        update_moving_average(  extractor_ema,  extractor, moving_average_decay)
        
        current_batch +=1
        if current_batch % print_every == 0:
            current_learning_rate = get_lr( optimizer )[0]
            LOG( "[current_batch: %05d] loss: %.3f, learning rate: %f"%( current_batch, running_loss/print_every,  current_learning_rate  ) )
            print( "[current_batch: %05d] loss: %.3f, learning rate: %f"%( current_batch, running_loss/print_every, current_learning_rate  ) )
            os.system( "nvidia-smi > %s/gpu_usage.log"%( log_folder ) )
            running_loss = 0

            if current_learning_rate < 1e-6:
                print("No progress is being made, stop training!")
                sys.exit(0)
        
        if validate_every != 0 and  current_batch % validate_every == 0:
            print("Starting validation ...")
            LOG("Starting validation ...")
            # validation
            val_score_list = []
            with torch.no_grad():
                for batch in tqdm(val_data_loader):
                    val_score_list += validation_iteration(batch)

            val_rouge1, val_rouge2, val_rougeL = list( zip( *val_score_list ) )

            avg_val_rouge1 = np.mean( val_rouge1 )
            avg_val_rouge2 = np.mean( val_rouge2 )
            avg_val_rougeL = np.mean( val_rougeL )
            print("val: %.4f, %.4f, %.4f"%(avg_val_rouge1, avg_val_rouge2, avg_val_rougeL))
            LOG("val: %.4f, %.4f, %.4f"%(avg_val_rouge1, avg_val_rouge2, avg_val_rougeL))
            # scheduler.step( (avg_val_rouge1 + avg_val_rouge2 +avg_val_rougeL)/3 )

        if  current_batch % save_every == 0:  
            save_model(  { 
                "current_batch": current_batch,
                "local_sentence_encoder": local_sentence_encoder_ema,
                "local_section_encoder": local_section_encoder_ema,
                "graph_encoder": graph_encoder_ema,
                "global_context_encoder": global_context_encoder_ema,
                "extraction_context_decoder":extraction_context_decoder_ema,
                "extractor": extractor_ema,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
                } , model_folder+"/model_batch_%d.pt"%(current_batch), max_to_keep = 100 )


