pfile_memsum_pred = './test_pred_memsum.json'
pfile_gosum_pred = './test_pred_go.json'
saved_data = './human_rate.json'
tag_num = 20

import json
import ipdb
import os
import random

data_memsum, data_gosum = [], []

def showline(nums, chars='#'):
    print(chars * nums)

def vis_gold(sents):
    showline(100)
    print("#  The gold summary sents are: \n")
    for i, sent in enumerate(sents):
        print('(' + str(i) + ')' + ' ' + sent)
        # print()
    print()
    print()

def vis_pred(sents1, sents2):
    showline(100)
    print("#  Two models both output those sents: \n")
    share_sents = list(set(sents1).intersection(set(sents2)))
    for i, sent in enumerate(share_sents):
        print('(' + str(i) + ')' + ' ' + sent)
    print()

    showline(100)
    print("#  The unique extracted sents for model 1 are: \n")
    for i, sent in enumerate(sents1):
        if sent not in share_sents:
            print('(' + str(i) + ')' + ' ' + sent)
            print()

    showline(100)
    print("#  The unique extracted sents for model 2 are: \n")
    for i, sent in enumerate(sents2):
        if sent not in share_sents:
            print('(' + str(i) + ')' + ' ' + sent)
            print()
    showline(100)


with open(pfile_memsum_pred, 'r') as of:
    for line in of:
        data = json.loads(line)
        data_memsum.append(data)

with open(pfile_gosum_pred, 'r') as of:
    for line in of:
        data = json.loads(line)
        data_gosum.append(data)

def avg_sum_len(data):
    lens = []
    for ele in data:
        lens.append(len(ele['pred']))
    print(sum(lens)/len(lens))


# avg_sum_len(data_memsum)
# ipdb.set_trace()

rate_results = []
tagged_idx = []

try:
    with open(saved_data, 'r') as of:
        for line in of:
            data = json.loads(line)
            rate_results.append(data)
            tagged_idx.append(data['idx'])
except: print('First time to tag..')


assert len(data_memsum) == len(data_gosum)

count = 0

# select_idx = []
dis_map = {'111':'222', '222':'111', '333':'333'}

for idx in range(len(data_memsum)):
    if idx in tagged_idx: continue
    if data_memsum[idx]['pred'] == data_gosum[idx]['pred']: continue

    print('You have tagged %d/%d results ! \n'%(len(rate_results),tag_num))

    vis_gold(data_memsum[idx]['gold'])
    assert data_memsum[idx]['gold'] == data_gosum[idx]['gold']

    disrupt = random.randint(0, 1)
    # disrupt = 0
    # ipdb.set_trace()
    if disrupt == 0:
        vis_pred(data_memsum[idx]['pred'], data_gosum[idx]['pred'])
    else:
        vis_pred(data_gosum[idx]['pred'], data_memsum[idx]['pred'])

    better_idx_rd = input("Please input which is less redundancy: ")
    while better_idx_rd not in ['111', '222', '333']:
        better_idx_rd = input("Please input which is less redundancy: ")

    better_idx_cv = input("Please input which is more coverage: ")
    while better_idx_cv not in ['111', '222', '333']:
        better_idx_cv = input("Please input which is more coverage: ")

    if disrupt == 1:
        better_idx_rd = dis_map[better_idx_rd]
        better_idx_cv = dis_map[better_idx_cv]


    rate_results.append({
        'idx': idx,
        'redundancy': better_idx_rd,
        'coverage': better_idx_cv
    })
    count += 1
    os.system('clear')

    with open(saved_data, 'w', encoding='utf-8') as wf:
        for item in rate_results:
            json.dump(item, wf)
            wf.write('\n')

    if count == tag_num:
        print('Thank you very much for your help, the tagging is over !!!')
