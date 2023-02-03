import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import ipdb
import dgl
import copy

from tqdm import tqdm


## the corpus has been preprocessed, so here only lower is needed
## all digits are kept, since sent2vec unigram embedding has digit embedding
## no stemming, no lemmatization
class SentenceTokenizer:
    def __init__(self ):
        pass
    def tokenize(self, sen ):
        return sen.lower()

class Vocab:
    def __init__(self, words, eos_token = "<eos>", pad_token = "<pad>", unk_token = "<unk>" ):
        self.words = words
        self.index_to_word = {}
        self.word_to_index = {}
        for idx in range( len(words) ):
            self.index_to_word[ idx ] = words[idx]
            self.word_to_index[ words[idx] ] = idx
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_index = self.word_to_index[self.eos_token]
        self.pad_index = self.word_to_index[self.pad_token]

        self.tokenizer = SentenceTokenizer()   

    def index2word( self, idx ):
        return self.index_to_word.get( idx, self.unk_token)
    def word2index( self, word ):
        return self.word_to_index.get( word, -1 )
    # The sentence needs to be tokenized 
    def sent2seq( self, sent, max_len = None , tokenize = True):
        if tokenize:
            sent = self.tokenizer.tokenize(sent)
        seq = []
        for w in sent.split():
            if w in self.word_to_index:
                seq.append( self.word2index(w) )
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len -1]
                seq.append( self.eos_index )
            else:
                seq.append( self.eos_index )
                seq += [ self.pad_index ] * ( max_len - len(seq) )
        return seq
    def seq2sent( self, seq ):
        sent = []
        for i in seq:
            if i == self.eos_index or i == self.pad_index:
                break
            sent.append( self.index2word(i) )
        return " ".join(sent)

class ExtractionTrainingDataset(Dataset):
    def __init__( self,  corpus, vocab , max_seq_len , max_doc_len  ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.max_sec_len = 50
        ## corpus is a list 
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

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



    
    def __getitem__( self, idx ):

        doc_data = self.corpus[idx]
        sentences = doc_data["text"]
        indices = doc_data["indices"]
        scores = np.array( doc_data["score"] )
        summary = doc_data["summary"]
        snames = doc_data['section_names']
        sbelong = doc_data['section_belong']

        num_sentences_in_doc = len( sentences )
        num_sections_in_doc = len( snames )

        ### This is for RL training
        rand_idx = np.random.choice( len(indices) )
        valid_sen_idxs = np.array( indices[ rand_idx ] )

        np.random.shuffle( valid_sen_idxs )
        # G = self.createGraph(num_sentences_in_doc, num_sections_in_doc, sbelong, valid_sen_idxs)
        # ipdb.set_trace()

        valid_sen_idxs = valid_sen_idxs[ valid_sen_idxs < num_sentences_in_doc ]
        selected_y_label = np.zeros( num_sentences_in_doc )
        selected_y_label[ valid_sen_idxs ] = 1
        selected_score = scores[ rand_idx ]
        

        valid_sen_idxs = valid_sen_idxs[:self.max_doc_len]
        valid_sen_idxs = np.array(valid_sen_idxs.tolist() + [-1] * ( self.max_doc_len - len(valid_sen_idxs)))


        if num_sentences_in_doc > self.max_doc_len:
            selected_y_label = selected_y_label[:self.max_doc_len]
            sentences = sentences[:self.max_doc_len]
            sbelong = sbelong[:self.max_doc_len]
            num_sentences_in_doc = self.max_doc_len
        else:
            selected_y_label = np.array( selected_y_label.tolist() + [0] * (self.max_doc_len - num_sentences_in_doc) )
            sentences += [""] * ( self.max_doc_len - num_sentences_in_doc )
        
        doc_mask = np.array(  [ True if sen.strip() == "" else False for sen in  sentences   ]  )
        seqs = [  self.vocab.sent2seq( sen, self.max_seq_len ) for sen in sentences ]
        seqs = np.asarray( seqs )
        sbelong = np.array(sbelong)


        if num_sections_in_doc > self.max_sec_len:
            snames = snames[:self.max_sec_len]
            num_sections_in_doc = self.max_sec_len
            for i_ in range(len(sbelong)):
                if sbelong[i_] >= self.max_sec_len: sbelong[i_] = self.max_sec_len - 1
        else:
            snames += [""] * ( self.max_sec_len - num_sections_in_doc )

        sec_mask = np.array(  [ True if sec.strip() == "" else False for sec in  snames   ]  )
        seqs_sec = [ self.vocab.sent2seq( sname, self.max_seq_len ) for sname in snames ]
        seqs_sec = np.asarray( seqs_sec )

        G = self.createGraph(num_sentences_in_doc, num_sections_in_doc, sbelong)


        summary = summary[:self.max_doc_len]
        if len(summary) < self.max_doc_len:
            summary = summary + [""] * ( self.max_doc_len - len(summary) )
        summary_seq = []
        for summary_sen in summary:
            summary_seq.append( np.array( self.vocab.sent2seq( summary_sen, self.max_seq_len ) ) )
        summary_seq = np.asarray(summary_seq)
        # ipdb.set_trace()
        
        return seqs, doc_mask, selected_y_label, selected_score, summary_seq, valid_sen_idxs, seqs_sec, sec_mask, sbelong, G, num_sentences_in_doc, num_sections_in_doc


class ExtractionValidationDataset(Dataset):
    def __init__( self,  corpus, vocab , max_seq_len , max_doc_len  ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.max_sec_len = 50
        ## corpus is a list 
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)
    
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
    
    def __getitem__( self, idx ):

        doc_data = self.corpus[idx]
        sentences = doc_data["text"]
        summary = doc_data["summary"]
        snames = doc_data['section_names']
        sbelong = doc_data['section_belong']

        num_sentences_in_doc = len( sentences )
        num_sections_in_doc = len( snames )

        if num_sentences_in_doc > self.max_doc_len:
            sentences = sentences[:self.max_doc_len]
            sbelong = sbelong[:self.max_doc_len]
            num_sentences_in_doc = self.max_doc_len
        else:
            sentences += [""] * ( self.max_doc_len - num_sentences_in_doc )
            
        doc_mask = np.array(  [ True if sen.strip() == "" else False for sen in  sentences   ]  )
            
        seqs = [  self.vocab.sent2seq( sen, self.max_seq_len ) for sen in sentences ]
        seqs = np.asarray( seqs )
        sbelong = np.array(sbelong)


        if num_sections_in_doc > self.max_sec_len:
            snames = snames[:self.max_sec_len]
            num_sections_in_doc = self.max_sec_len
            for i_ in range(len(sbelong)):
                if sbelong[i_] >= self.max_sec_len: sbelong[i_] = self.max_sec_len - 1
        else:
            snames += [""] * ( self.max_sec_len - num_sections_in_doc )

        sec_mask = np.array(  [ True if sec.strip() == "" else False for sec in  snames   ]  )
        seqs_sec = [ self.vocab.sent2seq( sname, self.max_seq_len ) for sname in snames ]
        seqs_sec = np.asarray( seqs_sec )

        summary = summary[:self.max_doc_len]
        if len(summary) < self.max_doc_len:
            summary = summary + [""] * ( self.max_doc_len - len(summary) )
        summary_seq = []
        for summary_sen in summary:
            summary_seq.append( np.array( self.vocab.sent2seq( summary_sen, self.max_seq_len ) ) )
        summary_seq = np.asarray(summary_seq)

        G = self.createGraph(num_sentences_in_doc, num_sections_in_doc, sbelong)

        return seqs, doc_mask, summary_seq, seqs_sec, sec_mask, sbelong, G, num_sentences_in_doc, num_sections_in_doc




def collate_fn(samples):
    '''
    :param batch: G, index, pid, idx, seg, clss, mask
    '''
    seqs_, doc_mask_, selected_y_label_, selected_score_, summary_seq_, valid_sen_idxs_, seqs_sec_, sec_mask_, sb_, G_, ns_, nc_ = map(list, zip(*samples))
    # graph_lists, index, pid = map(list, zip(*samples))
    # batched_graph_seq, batched_index_seq, batched_pid_seq = [], [], []
    seqs_b = torch.tensor(np.array(seqs_))
    doc_mask_b = torch.tensor(np.array(doc_mask_))
    selected_y_label_b = torch.tensor(np.array(selected_y_label_))
    selected_score_b = torch.tensor(np.array(selected_score_))
    summary_seq_b = torch.tensor(np.array(summary_seq_))
    valid_sen_idxs_b = torch.tensor(np.array(valid_sen_idxs_))
    seqs_sec_b = torch.tensor(np.array(seqs_sec_))
    sec_mask_b = torch.tensor(np.array(sec_mask_))
    ns_b = torch.tensor(np.array(ns_))
    nc_b = torch.tensor(np.array(nc_))
    # sb_b = np.array(sb_)
    G_b = dgl.batch([g_ for g_ in G_])
    # print(len(G_))
    # print(len(seqs_))

    # bs = len(glist_)
    return seqs_b, doc_mask_b, selected_y_label_b, selected_score_b, summary_seq_b, valid_sen_idxs_b, seqs_sec_b, sec_mask_b, sb_, G_b, ns_b, nc_b
    # return batched_graph_seq, (batched_index_seq, batched_pid_seq)


def collate_fn_valid(samples):
    '''
    :param batch: G, index, pid, idx, seg, clss, mask
    '''
    seqs_, doc_mask_, summary_seq_, seqs_sec_, sec_mask_, sb_, G_, ns_, nc_ = map(list, zip(*samples))
    # graph_lists, index, pid = map(list, zip(*samples))
    # batched_graph_seq, batched_index_seq, batched_pid_seq = [], [], []
    seqs_b = torch.tensor(np.array(seqs_))
    doc_mask_b = torch.tensor(np.array(doc_mask_))
    summary_seq_b = torch.tensor(np.array(summary_seq_))
    seqs_sec_b = torch.tensor(np.array(seqs_sec_))
    sec_mask_b = torch.tensor(np.array(sec_mask_))
    # sb_b = np.array(sb_)
    ns_b = torch.tensor(np.array(ns_))
    nc_b = torch.tensor(np.array(nc_))
    # sb_b = np.array(sb_)
    G_b = dgl.batch([g_ for g_ in G_])

    # bs = len(glist_)
    return seqs_b, doc_mask_b, summary_seq_b, seqs_sec_b, sec_mask_b, sb_, G_b, ns_b, nc_b
    # return batched_graph_seq, (batched_index_seq, batched_pid_seq)
