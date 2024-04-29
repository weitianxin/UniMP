import json,os
from PIL import Image
import numpy as np
from collections import defaultdict

data_name="all"
process_dir = f"."
save_name="denoise"
img_folder=data_name
num_items=41345
existing_imgs = [int(_.split(".")[0]) for _ in os.listdir(data_name)]
missing_imgs = []
for img_id in existing_imgs:
    try:
        img_ = Image.open(os.path.join(img_folder, f"{img_id}.jpg")).convert("RGB")
    except:
        missing_imgs.append(img_id)
print(missing_imgs)
print(f"{len(missing_imgs)} are broken")
if not len(missing_imgs):
    exit()
missing_imgs_dict = {img_id:0 for img_id in missing_imgs}
good_imgs = set(list(range(num_items)))-set(missing_imgs)
good_imgs = list(good_imgs)
good_imgs_dict = {img_id:0 for img_id in good_imgs}
with open(f"train_Office Products.json") as f:
    training_sequences = json.load(f)
with open(f"eval_Office Products.json") as f:
    eval_sequences = json.load(f)
with open(f"new_users.json") as f:
    test_sequences = json.load(f)
training_sequences.update(eval_sequences)
training_sequences.update(test_sequences)
with open(f"meta_all.json") as f:
    meta_data = json.load(f)
for user, full_seqs in training_sequences.items():
    pop_indexs = []
    new_full_seqs = [full_seq for i, full_seq in enumerate(full_seqs) if full_seq[0] in good_imgs_dict]
    training_sequences[user] = new_full_seqs


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
training_sequences = filter_Kcore(training_sequences, user_core=10, item_core=5)
asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))
meta_data = {asin: meta_data[str(asin)] for asin in asin_set}
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

if not os.path.exists(f"{process_dir}"):
    os.mkdir(f"{process_dir}")
with open(f'{process_dir}/full_{save_name}.json', 'w') as f:
    json.dump(new_data, f)
with open(f'{process_dir}/train_{save_name}.json', 'w') as f:
    json.dump(train_data, f)
with open(f'{process_dir}/eval_{save_name}.json', 'w') as f:
    json.dump(eval_data, f)
with open(f'{process_dir}/test_{save_name}.json', 'w') as f:
    json.dump(test_data, f)
with open(f'{process_dir}/meta_{save_name}.json', 'w') as f:
    json.dump(new_meta_data, f)

if not os.path.exists(f"{process_dir}/{save_name}"):
    os.mkdir(f"{process_dir}/{save_name}")
from tqdm import tqdm
for img_id in tqdm(asin_set):
    new_img_id = asin2id[img_id]
    img = Image.open(f"{data_name}/{img_id}.jpg").convert("RGB")
    img.save(f"{process_dir}/{save_name}/{new_img_id}.jpg")
    # os.system(f"mv {data_name}/{img_id}.jpg {process_dir}/{save_name}/{new_img_id}.jpg")
