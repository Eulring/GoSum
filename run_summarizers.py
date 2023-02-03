
from src.GoSum.model import LocalSentenceEncoder as LocalSentenceEncoder_GoSum
from src.GoSum.model import GlobalContextEncoder as GlobalContextEncoder_GoSum
from src.GoSum.model import ExtractionContextDecoder as ExtractionContextDecoder_GoSum
from src.GoSum.model import Extractor as Extractor_GoSum
from src.GoSum.model2 import SGraph as GraphEncoder_GoSum
from src.GoSum.datautils import Vocab as Vocab_GoSum
from src.GoSum.datautils import SentenceTokenizer as SentenceTokenizer_GoSum





import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
import torch
import numpy as np

from tqdm import tqdm
import json
import ipdb
import dgl



class ExtractiveSummarizer_GoSum:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =500, max_doc_len = 100  ):
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_GoSum( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_GoSum( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.local_section_encoder = LocalSentenceEncoder_GoSum( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.graph_encoder = GraphEncoder_GoSum( embed_dim )
        self.global_context_encoder = GlobalContextEncoder_GoSum( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_GoSum( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_GoSum( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = "cpu" )
        self.local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        self.local_section_encoder.load_state_dict( ckpt["local_section_encoder"] )
        self.graph_encoder.load_state_dict( ckpt["graph_encoder"] )
        self.global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        self.extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        self.extractor.load_state_dict(ckpt["extractor"])
        
        self.device =  torch.device( "cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu"  )        
        self.local_sentence_encoder.to(self.device)
        self.local_section_encoder.to(self.device)
        self.graph_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_GoSum()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set

    def createGraph(self, N_sent, N_sec, sbelong):
        G = dgl.DGLGraph()

        G.add_nodes(N_sent) # add sentence nodes
        G.ndata["unit"] = torch.ones(N_sent)
        G.ndata["dtype"] = torch.ones(N_sent)
        sentid2nid = [i for i in range(N_sent)]

        G.add_nodes(N_sec) # add section nodes
        G.ndata["unit"][N_sent:] = torch.ones(N_sec) * 2
        G.ndata["dtype"][N_sent:] = torch.ones(N_sec) * 2
        secid2nid = [i + N_sent for i in range(N_sec)]

        G.add_nodes(1) # add output nodes
        G.ndata["unit"][N_sent + N_sec:] = torch.ones(1) * 3
        G.ndata["dtype"][N_sent + N_sec:] = torch.ones(1) * 3
        outid2nid = [i + N_sent + N_sec for i in range(1)]

        G.add_nodes(N_sent) # add global sentence nodes
        G.ndata["unit"][N_sent + N_sec + 1:] = torch.ones(N_sent) * 4
        G.ndata["dtype"][N_sent + N_sec + 1:] = torch.ones(N_sent) * 4
        gsentid2nid = [i + N_sent + N_sec + 1 for i in range(N_sent)]

        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(N_sent):
            sent_nid = sentid2nid[i]
            sec_nid = secid2nid[sbelong[i]]

            G.add_edge(sec_nid, sent_nid, data={"dtype": torch.Tensor([1])})
            G.add_edge(sent_nid, sec_nid, data={"dtype": torch.Tensor([1])})

            for j in range(N_sec):
                G.add_edge(secid2nid[j], gsentid2nid[i], data={"dtype": torch.Tensor([1])})

        for i in range(N_sec):
            for j in range(N_sec):
                if i != j: G.add_edge(secid2nid[i], secid2nid[j], data={"dtype": torch.Tensor([1])})

        return G


    def extract( self, document_batch, sname_batch, sbelong_batch, p_stop_thres = 0.7, ngram_blocking = False, ngram = 3, return_sentence_position = False, return_sentence_score_history = False, max_extracted_sentences_per_document = 4 ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        assert len(document_batch) == 1
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        tokenized_sections_batch = []
        section_length_list = []
        for sections in sname_batch:
            tokenized_sections = []
            for sec in sections:
                tokenized_sec = self.sentence_tokenizer.tokenize( sec )
                tokenized_sections.append( tokenized_sec )
                section_length_list.append( len(tokenized_sec.split()) )
            tokenized_sections_batch.append( tokenized_sections )

        # ipdb.set_trace()
        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        max_section_length = 50
        ## convert to sequence
        seqs = []
        doc_mask = []
        sbelongs = []
        
        for idx, document in enumerate(tokenized_document_batch):
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
                sbelong = sbelong_batch[idx][:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [""] * ( max_document_length -  len(document) )
                sbelong = sbelong_batch[idx]

            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
            sbelongs.append(sbelong)

        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        # process sections
        seqs_sec = []
        sec_mask = []
        for sections in tokenized_sections_batch:
            if len(sections) > max_section_length: 
                sections = sections[:max_section_length]
            else:
                sections = sections + [""] * ( max_section_length - len(sections) )
        
            sec_mask.append( [ 1 if sec.strip() == "" else 0 for sec in  sections   ] )
            sec_sequences = []
            for sec in sections:
                seq_sec = self.vocab.sent2seq( sec, max_sentence_length )
                sec_sequences.append(seq_sec)
            seqs_sec.append(sec_sequences)


        seqs_sec = np.asarray(seqs_sec)
        sec_mask = np.asarray(sec_mask) == 1
        seqs_sec = torch.from_numpy(seqs_sec).to(self.device)
        sec_mask = torch.from_numpy(sec_mask).to(self.device)

        # build graph
        num_sent = min(self.max_doc_len , len(document_batch[0]))
        num_sec = min(50, len(sname_batch[0]))
        Glist = [self.createGraph( num_sent, num_sec, sbelongs[0] )]
        G = dgl.batch([g_ for g_ in Glist])
        G = G.to(self.device)

    
        extracted_sentences = []
        sentence_score_history = []
        p_stop_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )

            num_sections = seqs_sec.size(1)
            sec_embed = self.local_section_encoder( seqs_sec.view(-1, seqs_sec.size(2) )  )
            sec_embed = sec_embed.view( -1, num_sections, sec_embed.size(1) )

            sent_state = torch.zeros((num_sent, sen_embed.shape[2])).to(self.device)
            sec_state = torch.zeros((num_sec, sec_embed.shape[2])).to(self.device)
            sent_state[:num_sent] = sen_embed[0][:num_sent]
            sec_state[:num_sec] = sec_embed[0][:num_sec]
            
            global_sen_embed_, global_sec_embed_, global_gsen_embed_ = self.graph_encoder(G, sent_state, sec_state, sent_state)
            
            global_sen_embed = torch.zeros_like(sen_embed)
            global_sec_embed = torch.zeros_like(sec_embed)
            global_gsen_embed = torch.zeros_like(sen_embed)

            global_sen_embed[0][:num_sent] = global_sen_embed_[:num_sent]
            global_gsen_embed[0][:num_sent] = global_gsen_embed_[:num_sent]
            global_sec_embed[0][:num_sec] = global_sec_embed_[:num_sec]

            relevance_embed = self.global_context_encoder( global_gsen_embed, doc_mask  )
            
            
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []


            # sent_state = torch.zeros((num_documents, sen_embed.shape[2])).to(self.device)
            # sec_state = torch.zeros((num_sections, sec_embed.shape[2])).to(self.device)
            # sent_state[:num_documents] = sen_embed[0][:num_documents]
            # sec_state[:num_sections] = sec_embed[0][:num_sections]
            # global_sen_embed, global_sec_embed = self.graph_encoder(G, sent_state, sec_state)

        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_sec_embed = sec_embed[doc_i:doc_i+1]
                current_sbelong = sbelongs[doc_i]
                # ipdb.set_trace()
                sec_embed_per_sent = torch.zeros_like(current_sen_embed)
                sec_embed_per_sent[doc_i][:len(current_sbelong)] = global_sec_embed[doc_i][current_sbelong]
                # ipdb.set_trace()

                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()
                sentence_score_history_for_doc_i = []
                p_stop_history_for_doc_i = []
                
                for step in range( max_extracted_sentences_per_document+1 ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( global_sen_embed, current_remaining_mask, current_extraction_mask  )

                    p, p_stop, _ = self.extractor( global_sen_embed, sec_embed_per_sent, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    p_stop = p_stop.unsqueeze(1)
                    
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy() )

                    p_stop_history_for_doc_i.append(  p_stop.squeeze(1).item() )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = sorted_sen_indices[0]
                    
                    extracted = False
                    for sen_i in sorted_sen_indices:
                        sen_i = sen_i.item()
                        if sen_i< len(document_batch[doc_i]):
                            sen = document_batch[doc_i][sen_i]
                        else:
                            break
                        sen_ngrams = self.get_ngram( sen.lower().split(), ngram )
                        if not ngram_blocking or len( extracted_sen_ngrams &  sen_ngrams ) < 1:
                            extracted_sen_ngrams.update( sen_ngrams )
                            extracted = True
                            break
                                        
                    if stop or step == max_extracted_sentences_per_document or not extracted:
                        extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])    ] )
                        extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])  ]  )
                        break
                    else:
                        current_hyps.append(sen_i)
                        current_extraction_mask_np[0, sen_i] = True
                        current_remaining_mask_np[0, sen_i] = False

                sentence_score_history.append(sentence_score_history_for_doc_i)
                p_stop_history.append( p_stop_history_for_doc_i )

        # if return_sentence_position:
        #     return extracted_sentences, extracted_sentences_positions 
        # else:
        #     return extracted_sentences

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results+=[sentence_score_history , p_stop_history ]
        if len(results) == 1:
            results = results[0]
        # ipdb.set_trace()
        return results







