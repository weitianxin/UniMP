import json
import numpy as np
data_name="all"
def keep_exp_json(split="train"):
    keep_exp_data={}
    ori_length=[]
    new_length=[]
    if split=="train":
        thresh_num = 6
    elif split=="eval":
        thresh_num = 7
    elif split=="test":
        thresh_num = 8
    with open(f'{split}_{data_name}.json') as f:
        data = json.load(f)
        for key, full_seq in data.items():
            new_full_seq = []
            for full_item in full_seq:
                if full_item[1]!="":
                    new_full_seq.append(full_item)
            if len(new_full_seq)>=thresh_num:
                keep_exp_data[key]=new_full_seq
                ori_length.append(len(full_seq))
                new_length.append(len(new_full_seq))
    print(f"split {split}:",np.mean(ori_length), np.mean(new_length))
    with open(f'{split}_{data_name}_exp.json', 'w') as f:
        json.dump(keep_exp_data, f)
keep_exp_json("train")
keep_exp_json("eval")
keep_exp_json("test")

