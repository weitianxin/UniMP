from collections import defaultdict
import json
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import random, os
training_sequences = defaultdict(list)
item2attr = {}
num_line = 0

def get_time(date):    
    return datetime.datetime.strptime(date,"%Y-%m-%d").timestamp()
# data = pd.read_csv("transactions_train.csv")
# print(len(data))
# for index, row in tqdm(data.iterrows()):
#     item_id = str(row["article_id"])
#     user_id = str(row["customer_id"])
#     # time = row["t_dat"]
#     time = get_time(row["t_dat"])
#     training_sequences[user_id].append([time, item_id])

# def post_process(sequences):
#     length = 0
#     for user, sequence in tqdm(sequences.items()):
#         # sequence = sorted(sequence, key=lambda date: get_time(date[0]))
#         sequences[user] = [ele[1:] for ele in sorted(sequence)]
#         length += len(sequences[user])

#     print(f'Averaged length: {length/len(sequences)}')

#     return sequences
# training_sequences = post_process(training_sequences)

# with open('data_sort.json', 'w') as f:
#     json.dump(training_sequences, f)
def fill(num):
    s = str(num)
    s = "0"*(10-len(s))+s
    return s
with open('data_sort.json') as f:
    training_sequences = json.load(f)

image_dirs = os.listdir("images")
existing_images = []
for dir in image_dirs:
    for aid in os.listdir(f"images/{dir}"):
        existing_images.append(aid.split(".")[0])
print(len(existing_images))

item = pd.read_csv("articles.csv")
for index, row in item.iterrows():
    item_id = str(fill(row["article_id"]))
    item2attr[item_id] = [row["prod_name"], row["graphical_appearance_name"], row["perceived_colour_value_name"], row["section_name"], row["detail_desc"]]
# remove duplicates
for user, items in training_sequences.items():
    items = [_[0] for _ in items]
    new_items = list(set([item for item in items]))
    new_items.sort(key=items.index)
    training_sequences[user] = [[item] for item in new_items]
item_sequences = defaultdict(list)
for user, items in training_sequences.items():
    for item in items:
        item = fill(item[0])
        item_sequences[item].append(user)
items_with_image = set(item_sequences.keys())&set(existing_images)
print(len(items_with_image))
print(set([str(_)[:3] for _ in list(items_with_image)]))
item_sequences = {item:item_sequences[item] for item in items_with_image}

training_sequences = defaultdict(list)
for item, users in item_sequences.items():
    for user in users:
        training_sequences[user].append([item])
# print(len(item_sequences))
keys = list(training_sequences.keys())
random.seed(42)
random.shuffle(keys)
select_keys = keys[:30000]
length = 0
for user, sequence in tqdm(training_sequences.items()):
    # sequence = sorted(sequence, key=lambda date: get_time(date[0]))
    length += len(training_sequences[user])
print(f'Averaged length: {length/len(training_sequences)}')
training_sequences = {key: training_sequences[key] for key in select_keys}
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

print(len(training_sequences))
training_sequences = filter_Kcore(training_sequences, 10, 10)
asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
item2attr = {asin: item2attr[asin] for asin in asin_set}
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))
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
import os, copy
from PIL import Image
process_dir = "."
data_name = "hm"
if not os.path.exists(f"{process_dir}/{data_name}"):
    os.mkdir(f"{process_dir}/{data_name}")
for old_value, value in tqdm(zip(keys, values)):
    prefix = str(old_value)[:3]
    in_filepath=f"{process_dir}/images/{prefix}/{old_value}.jpg"
    out_filepath=f"{process_dir}/{data_name}/{value}.jpg"
    im= Image.open(in_filepath).convert('RGB')
    im.save(out_filepath)

new_data, new_meta_data = copy.deepcopy(training_sequences), {}
for user, values in training_sequences.items():
    for i, value in enumerate(values):
        new_data[user][i][0] = asin2id[value[0]]
for asin, attr in item2attr.items():
    id = asin2id[asin]
    new_meta_data[id] = attr

import random
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
with open(f'{process_dir}/full_users.json', 'w') as f:
    json.dump(new_data, f)
with open(f'{process_dir}/train_users.json', 'w') as f:
    json.dump(train_data, f)
with open(f'{process_dir}/eval_users.json', 'w') as f:
    json.dump(eval_data, f)
with open(f'{process_dir}/test_users.json', 'w') as f:
    json.dump(test_data, f)
with open(f'{process_dir}/meta_{data_name}.json', 'w') as f:
    json.dump(new_meta_data, f)
