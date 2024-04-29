import pickle, json
attr = pickle.load(open("augmented_attribute_dict","rb"))
# attr = pickle.load(open("augmented_sample_dict","rb"))


print(attr[0], min(list(attr.keys())), max(list(attr.keys())))
path = "."
train_file = path + '/train.json'
val_file = path + '/val.json' 
test_file = path + '/test.json'
n_users, n_items = 0, 0
n_train, n_test, n_val = 0, 0, 0
neg_pools = {}

exist_users = []
train = json.load(open(train_file))
test = json.load(open(test_file))
val = json.load(open(val_file))
print(len(train), len(test), len(val))
from collections import defaultdict
user2item = defaultdict(list)
for item, users in val.items():
    for user in users:
        user2item[user].append(item)
print(len(user2item))
for uid, items in train.items():
    if len(items) == 0:
        continue
    uid = int(uid)
    exist_users.append(uid)
    n_items = max(n_items, max(items))
    n_users = max(n_users, uid)
    n_train += len(items)

for uid, items in test.items():
    uid = int(uid)
    try:
        n_items = max(n_items, max(items))
        n_test += len(items)
    except:
        continue

for uid, items in val.items():
    uid = int(uid)
    try:
        n_items = max(n_items, max(items))
        n_val += len(items)
    except:
        continue

n_items += 1
n_users += 1
print('n_users=%d, n_items=%d' % (n_users, n_items))
print('n_interactions=%d' % (n_train + n_test))
print('n_train=%d, n_test=%d, sparsity=%.5f' % (n_train, n_test, (n_train + n_test)/(n_users * n_items)))