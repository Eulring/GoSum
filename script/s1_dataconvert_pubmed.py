import json
import ipdb
import os
from tqdm import tqdm

splits = ['train', 'test', 'val']
original_pubmed_root = '../data/pubmed-dataset'
tar_file_root = '../data/pubmed-sec'

data_collect = {'train':[], 'test':[], 'val':[]}

for split in splits:
    file_name = os.path.join(original_pubmed_root, split + '.txt')

    with open(file_name, 'r') as of:
        lines = of.readlines()
        for line in lines:
            item = json.loads(line)
            ipdb.set_trace()
            summary, text, section_belong = [], [], []

            for sent in item['abstract_text']:
                sent_ = sent.replace('<S> ', '').replace(' </S>', '')
                summary.append(sent_)

            for i, section in enumerate(item['sections']):
                for sent in section:
                    text.append(sent)
                    section_belong.append(i)

            assert len(item['sections']) == len(item['section_names'])

            data_collect[split].append({
                'text': text,
                'summary': summary,
                'section_belong': section_belong,
                'section_names': item['section_names']
            })

for split in splits:
    fname = split + '.jsonl'
    tar_file_path = os.path.join(tar_file_root, fname)
    print(split)
    print(len(data_collect[split]))
    with open(tar_file_path, 'w', encoding='utf-8') as wf:
        for item in data_collect[split]:
            json.dump(item, wf)
            wf.write('\n')