import json
import ipdb

pfile_eval1 = './human_rate.json'
pfile_eval2 = './human_rate2.json'

eval_num = 50

emap = {'111':1, '222':2, '333':1.5}

cv_m1 = [[] for i in range(eval_num)]
cv_m2 = [[] for i in range(eval_num)]
rd_m1 = [[] for i in range(eval_num)]
rd_m2 = [[] for i in range(eval_num)]


rate_results = []
with open(pfile_eval1, 'r') as of:
    for line in of:
        data = json.loads(line)
        rate_results.append(data)

for i in range(eval_num):
    e1 = rate_results[i]
    rd = e1['redundancy']

    rd_m2[i].append(3 - emap[e1['redundancy']])
    rd_m1[i].append(emap[e1['redundancy']])

    cv_m2[i].append(3 - emap[e1['coverage']])
    cv_m1[i].append(emap[e1['coverage']])

rate_results = []
with open(pfile_eval2, 'r') as of:
    for line in of:
        data = json.loads(line)
        rate_results.append(data)

for i in range(eval_num):
    e1 = rate_results[i]
    rd = e1['redundancy']

    rd_m2[i].append(3 - emap[e1['redundancy']])
    rd_m1[i].append(emap[e1['redundancy']])

    cv_m2[i].append(3 - emap[e1['coverage']])
    cv_m1[i].append(emap[e1['coverage']])


def cal_rank(ranks):
    cnt, rank = 0, 0
    for ele in ranks:
        for e in ele:
            if e != 1.5:
                cnt += 1
                rank += e
    print(rank / cnt)

cal_rank(rd_m1)
cal_rank(rd_m2)

cal_rank(cv_m1)
cal_rank(cv_m2)

# ipdb.set_trace()
