import json
import numpy as np
import random
import numpy as np
data_name="all"
NUM_ITEMS=3
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
    
def gen_img_sel(split="train",item_set=None):
    img_sel_data={}
    with open(f'{split}_{data_name}.json') as f:
        data = json.load(f)
        
        for key, full_seq in data.items():
            p=np.random.random_sample()
            if p<0.6:
                new_full_seq = full_seq[:-1]
                items_cur_seq = [item[0] for item in full_seq]
                neg_items = list(item_set-set(items_cur_seq))
                negs = list(np.random.choice(neg_items, NUM_ITEMS-1, replace=False))
                items = [full_seq[-1][0]]
                negs.extend(items)
                np.random.shuffle(negs)
                labels=[]
                for i, item in enumerate(negs):
                    if item in items:
                        labels.append(i)
                target = [full_seq[-1], negs, labels]
                new_full_seq.append(target)
            else:
                new_full_seq = full_seq[:-2]
                items_cur_seq = [item[0] for item in full_seq]
                neg_items = list(item_set-set(items_cur_seq))
                negs = list(np.random.choice(neg_items, NUM_ITEMS-2, replace=False))
                items = [full_seq[-2][0], full_seq[-1][0]]
                negs.extend(items)
                np.random.shuffle(negs)
                labels=[]
                for i, item in enumerate(negs):
                    if item in items:
                        labels.append(i)
                target = [full_seq[-2], full_seq[-1], negs, labels]
                new_full_seq.append(target)
            img_sel_data[key]=new_full_seq
    with open(f'{split}_{data_name}_img_sel.json', 'w') as f:
        json.dump(img_sel_data, f, cls=NumpyEncoder)
item_set = set(range(22738))
gen_img_sel("train",item_set)
gen_img_sel("eval",item_set)
gen_img_sel("test",item_set)


        