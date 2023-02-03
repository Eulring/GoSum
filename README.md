# Dependency

Torch (our cuda version is 11.3)
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Install DGL
```bash
conda install -c dglteam dgl-cuda11.3
```

Download glove embedding from https://drive.google.com/drive/folders/1lrwYrrM3h0-9fwWCOmpRkydvmF6hmvmW

# Training 

## On sampled dataset
It takes lot of time to process the data from the original dataset, and also due to submission limitations, so we provide a sampled and processed data for you to run the program. (#train-1000 / #val-500 / #test-500)

step1: set correct path of glove embedding in /GoSum/src/GoSum/config/pubmed/training.config

step2: input training command:
```bash
cd ./src/GoSum
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_dist.py -config_file_path config/pubmed/training.config
```

## On full dataset
Download full PubMed and arXiv dataset from: https://github.com/armancohan/long-summarization.

go to script fold and follow steps below:

#### step1: data convert
```bash
cd ./script
python s1_dataconvert_pubmed.py
python s1_dataconvert_arxiv.py
```

#### step2: beam search to get oracle labels
```bash
sh create_dataset_faster.sh 0 119924 10000 ../data/pubmed-sec/train.jsonl ../data/pubmed-sec/processed/pro_train.jsonl 2 7 15 0.001 10000

sh create_dataset_faster.sh 0 203037 20000 ../data/arxiv-sec/train.jsonl ../data/arxiv-sec/processed/pro_train.jsonl 2 7 15 0.001 10000
```

#### step3: collect sample files

```bash
python s2_mergefiles.py \
 -folder ../data/pubmed-sec/processed \
 -prefix pro_train.jsonl \
 -save_name train.jsonl

python s2_mergefiles.py \
 -folder ../data/arxiv-sec/processed \
 -prefix pro_train.jsonl \
 -save_name train.jsonl
```


# Testing

```bash
python test.py -model_type GoSum -summarizer_model_path YOUR_MODEL_PATH -vocabulary_path YOUR_VOCAB_PATH -corpus_path ./data/pubmed/test_500.jsonl -gpu 3 -max_extracted_sentences_per_document 7 -p_stop_thres 0.6 -output_file results/results.txt  -max_doc_len 500 -max_seq_len 100
```

# Reference

If you have used this code or found our work helpful, please cite 
```
Bian, Junyi, et al. "GoSum: Extractive Summarization of Long Documents by Reinforcement Learning and Graph Organized discourse state." arXiv preprint arXiv:2211.10247 (2022).
```

Our model references __MemSum__ and __HSG__. I appreciate them making the code public, here is the Github link to their code.

```
Memsum: https://github.com/nianlonggu/MemSum
```

```
HSG: https://github.com/dqwang122/HeterSumGraph
```

