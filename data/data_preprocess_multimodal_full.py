import json
import pickle
from tqdm import tqdm
import random
import numpy as np
import os
from urllib import request

data_names = ["Baby", "Beauty", "Clothing Shoes and Jewelry",\
              "Toys and Games", "Sports and Outdoors", "Grocery and Gourmet Food"]
data_names = ["Beauty"]
# data_names = ["Tools and Home Improvement","Office Products"]
# data_names = ["Clothing Shoes and Jewelry"]
# data_names = ["Baby", "Beauty", "Clothing Shoes and Jewelry", "Toys and Games", "Sports and Outdoors", "Grocery and Gourmet Food"]


if len(data_names)>1:
    save_name = "all"
else:
    save_name = data_names[0]
user_core=8
item_core=5
process_dir = f"processed_filter_{user_core}_{save_name}"

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
# meta data extraction
def extract_meta(data_name, meta_data):
    data_name = "_".join(data_name.split(" "))
    print("Extract Meta", data_name)
    meta_path = f"meta_{data_name}.json"
    if not os.path.exists(meta_path):
        os.system(f"wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{data_name}.json.gz")
        os.system(f"gzip -d meta_{data_name}.json.gz")
    num1,num2,num3,num4,num5 = 0,0,0,0,0
    num_noimage=0
    with open(meta_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            dict_line = eval(line)
            attr_dict = {}
            if "imUrl" in dict_line:
                attr_dict["imUrl"] = dict_line["imUrl"]
                if "categories" in dict_line:
                    category = ' '.join(dict_line['categories'][0])
                    attr_dict['category'] = category
                else:
                    attr_dict['category'] = ""
                    num1+=1
                if "brand" in dict_line:
                    brand = dict_line['brand']
                    attr_dict['brand'] = brand
                else:
                    attr_dict['brand'] = ""
                    num2+=1
                if "title" in dict_line:
                    title = dict_line['title']
                    attr_dict['title'] = title
                else:
                    attr_dict['title'] = ""
                    num3+=1
                if "description" in dict_line:
                    des = dict_line['description']
                    attr_dict['description'] = des
                else:
                    attr_dict['description'] = ""
                    num4+=1
                if "price" in dict_line:
                    price = dict_line['price']
                    attr_dict['price'] = price
                else:
                    attr_dict['price'] = ""
                    num5+=1
                asin = dict_line["asin"]
                meta_data[asin] = attr_dict
    print(num_noimage,num1, num2, num3, num4,num5)
    return meta_data
meta_data={}
for data_name in data_names:
    meta_data=extract_meta(data_name, meta_data=meta_data)

from collections import defaultdict

def extract_interaction(data_name, sequences, asin_set):
    data_name = "_".join(data_name.split(" "))
    print("Extract Interactions", data_name)
    inter_path = f"reviews_{data_name}_5.json"
    if not os.path.exists(inter_path):
        os.system(f"wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{data_name}_5.json.gz")
        os.system(f"gzip -d reviews_{data_name}_5.json.gz")
    user_set, item_set, inter_num = set(), set(), 0
    exp_path = 'raw_data/reviews_{}.pickle'.format(data_name)
    if os.path.exists(exp_path):
        raw_explanations = load_pickle(exp_path)
        use_exp=True
    else:
        use_exp=False
    with open(inter_path,"r") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            
            dict_line = eval(line)
            # add explanation only for partial datasets
            if use_exp:
                raw_explanation = raw_explanations[i]
                assert dict_line['reviewerID'] == raw_explanation['user']
                assert dict_line['asin'] == raw_explanation['item']
                if 'sentence' in raw_explanation:
                    list_len = len(raw_explanation['sentence'])
                    selected_idx = random.randint(0, list_len-1)
                    explanation = raw_explanation['sentence'][selected_idx][2]
                else:
                    explanation = ""
            else:
                explanation = ""
            # add end
            user = dict_line['reviewerID']
            asin = dict_line['asin']
            time = dict_line['unixReviewTime']
            review = dict_line['reviewText']
            rate = dict_line['overall']
            summary = dict_line['summary']
            if asin in meta_data:
                sequences[user+'_'+data_name].append([time, asin, explanation, rate, summary, review])
                user_set.add(user)
                item_set.add(asin)
                inter_num += 1
                asin_set.add(asin)                
        print(f'Dataset: {data_name}, User: {len(user_set)}, Items: {len(item_set)}, Interaction numbers: {inter_num} asin_set: {len(asin_set)}')

    return sequences, asin_set


training_sequences = defaultdict(list)
asin_set = set()
for data_name in data_names:
    training_sequences, asin_set = extract_interaction(data_name, training_sequences, asin_set)
import copy
def post_process(sequences):
    length = 0
    for user, sequence in tqdm(sequences.items()):
        sequences[user] = [ele[1:] for ele in sorted(sequence)]
        length += len(sequences[user])

    print(f'Averaged length: {length/len(sequences)}')

    return sequences

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item[0]] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for full_item in user_items[user]:
                    item = full_item[0]
                    if item_count[item] < item_core:
                        item_user = [full_item[0]==item for full_item in user_items[user]]
                        index = np.where(item_user)[0][0]
                        user_items[user].pop(index)
                        # user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


training_sequences = post_process(training_sequences)
training_sequences = filter_Kcore(training_sequences, user_core=user_core, item_core=item_core)

asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))
meta_data = {asin: meta_data[asin] for asin in asin_set}
# reorder
asin2id={}
id=0
for user, values in training_sequences.items():
    asins = [value[0] for value in values]
    for asin in asins:
        asin2id.setdefault(asin, id)
        if asin2id[asin]==id:
            id+=1
keys = list(asin2id.keys())

values = list(asin2id.values())
import random, copy
old_values = copy.deepcopy(values)
random.seed(42)
random.shuffle(values)
for key, value in zip(keys, values):
    asin2id[key] = value
# resave images
import os

# from PIL import Image
# if not os.path.exists(f"{process_dir}/{data_name}"):
#     os.mkdir(f"{process_dir}/{data_name}")
# for old_value, value in tqdm(zip(old_values, values)):
#     in_filepath=f"{process_dir}/ori_{data_name}/{old_value}.jpg"
#     out_filepath=f"{process_dir}/{data_name}/{value}.jpg"
#     im= Image.open(in_filepath).convert('RGB')
#     im.save(out_filepath)

new_data, new_meta_data = copy.deepcopy(training_sequences), {}
for user, values in training_sequences.items():
    for i, value in enumerate(values):
        new_data[user][i][0] = asin2id[value[0]]
for asin, attr in meta_data.items():
    id = asin2id[asin]
    new_meta_data[id] = attr


keys = list(new_data.keys())
random.seed(42)
random.shuffle(keys)
num = int(len(keys)*0.8)
num1 = int(len(keys)*0.9)
train_keys, eval_keys, test_keys = keys[:num], keys[num: num1], keys[num1:]
train_data = {key: new_data[key] for key in train_keys}
test_data = {key: new_data[key] for key in test_keys}
eval_data = {key: new_data[key] for key in eval_keys}

# print(new_meta_data[4265])
# print(new_meta_data[3278])
# with open(f'{process_dir}/meta_{data_name}.json', 'w') as f:
#     json.dump(new_meta_data, f)
# exit()

if not os.path.exists(f"{process_dir}"):
    os.mkdir(f"{process_dir}")
with open(f'{process_dir}/users.json', 'w') as f:
    json.dump(new_data, f)
with open(f'{process_dir}/train_users.json', 'w') as f:
    json.dump(train_data, f)
with open(f'{process_dir}/eval_users.json', 'w') as f:
    json.dump(eval_data, f)
with open(f'{process_dir}/test_users.json', 'w') as f:
    json.dump(test_data, f)
with open(f'{process_dir}/meta_{save_name}.json', 'w') as f:
    json.dump(new_meta_data, f)

    
import requests

def down_save(url, image_name):
    r = requests.get(url, stream=True)
    with open(image_name, 'wb') as f:
        f.write(r.content)

with open(f'{process_dir}/meta_{save_name}.json') as f:
    meta_data = json.load(f)
if not os.path.exists(f"{process_dir}/{save_name}"):
    os.mkdir(f"{process_dir}/{save_name}")
for key, values in tqdm(meta_data.items()):
    imUrl = values["imUrl"]
    out_filepath=f"{process_dir}/{save_name}/{key}.jpg"
    try:
        down_save(imUrl, out_filepath)
    except:
        raise KeyError(imUrl)