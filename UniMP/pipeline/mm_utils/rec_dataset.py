# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


import base64
from io import BytesIO
import re
import contextlib
import os

from PIL import ImageFile
from torchvision import transforms

from .transforms import *
from .input_dataset import FileDataset

from .collate_rec import (
    collate_fn,
)

import os,json

label_map = {"entailment": 0, "not_entailment": 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.48145466, 0.4578275, 0.40821073]
FLAMINGO_STD = [0.26862954, 0.26130258, 0.27577711]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, args, is_test=False, supported_data_types=["seq_rec"], split="train", task="rec"):
        # super().__init__()
        self.split = split
        self.args = args
        self.task_name = task
        self.is_test = is_test
        self.tokenizer = args.tokenizer
        self.tasks = task
        self.use_semantic=args.use_semantic
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size
        self.supported_data_types = supported_data_types

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]
        

        # TODO: check if random augment is correct, especially for some questions related to colors.
        # self.patch_resize_transform = transforms.Compose(
        #     [
        #         RandomResize(scales),
        #         transforms.CenterCrop(args.patch_image_size),
        #         transforms.RandomHorizontalFlip(p=0.2),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
        #     ]
        # )
        if split=="train":
            self.patch_resize_transform = transforms.Compose(
                [
                    RandomResize(scales),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
                ]
            )
        else:
            self.patch_resize_transform = transforms.Compose(
                [
                    RandomResize(scales),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
                ]
            )
        self.folder_path = args.mmrec_path
        self.subset = args.subset
        
        self.img_folder = os.path.join(self.folder_path, self.subset)
        # ！！！
        # self.data_path = os.path.join(self.folder_path, f"{split}_{self.subset}.json")
        self.data_path = os.path.join(self.folder_path, f"{split}_users.json")
        # ！！！
        self.data_img_sel_path = os.path.join(self.folder_path, f"{split}_{self.subset}_img_sel.json")
        self.data_exp_path = os.path.join(self.folder_path, f"{split}_{self.subset}_exp.json")
        self.data_img_gen_path = os.path.join(self.folder_path, f"{split}_{self.subset}.json")
        self.retrieval_data_path = os.path.join(self.folder_path, f"search_merge_{split}.txt")
        self.data_search_path = self.data_path
        self.meta_path = os.path.join(self.folder_path, f"meta_{self.subset}.json")
        # img semantic id
        self.img_id_path = os.path.join(self.folder_path, f"img_id2semantic.json")
        
        # semantic id
        if args.use_semantic:
            self.len_semanticid = 3
            self.id_path = os.path.join(self.folder_path, f"id2semantic.json")
            with open(self.id_path) as f:
                self.id2semantic = json.load(f)
            
            
        # history length
        if self.subset=="all":
            if self.tasks=="img_gen":
                self.history_len = 2
            else:
                self.history_len = 5
        if self.subset=="netflix":
            self.history_len = 3
        if self.subset=="hm":
            self.history_len = 8
        with open(self.meta_path) as f:
            self.meta_data = json.load(f)
        if split=="train":
            if args.single_task:
                if self.tasks=="rec":
                    with open(self.data_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["rec"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="exp":
                    with open(self.data_exp_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["exp"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="img_sel":
                    with open(self.data_img_sel_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["img_sel"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="search":
                    with open(self.data_search_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["search"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="img_gen":
                    with open(self.retrieval_data_path) as f:
                        self.retrieval_data = json.load(f)
                    with open(self.img_id_path) as f:
                        self.img_id2semantic = json.load(f)
                    # pretrain!!!
                    # with open(self.meta_path) as f:
                    #     self.data = json.load(f)
                    with open(self.data_img_gen_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["img_gen"]*len(self.data)
                    self.seqs = self.retrieval_data
                self.keys = list(self.data.keys())
            elif type(task)==list:
                self.t2id={"img_sel":0, "search":1, "rec":2, "exp":3}
                with open(self.data_path) as f:
                    self.rec_data = json.load(f)
                with open(self.data_exp_path) as f:
                    self.exp_data = json.load(f)
                with open(self.data_img_sel_path) as f:
                    self.sel_data = json.load(f)
                with open(self.data_search_path) as f:
                    self.search_data = json.load(f)
                self.data = [self.sel_data, self.search_data, self.rec_data, self.exp_data]
                self.seqs, self.tasks = [], []
                n = len(task)
                for i, t in enumerate(task):
                    ind = self.t2id[t]
                    t_data = self.data[ind]
                    if i<n-1:
                        t_keys = list(t_data.keys())
                        np.random.shuffle(t_keys)
                        n_keys = int(0.25*len(t_keys))
                        cur_keys = t_keys[:n_keys]
                        cur_data = {key: t_data[key] for key in cur_keys}
                    else:
                        cur_data = t_data
                    cur_seq = list(cur_data.values())
                    self.seqs += cur_seq
                    self.tasks += [t]*len(cur_seq)
            else:
                with open(self.data_path) as f:
                    self.rec_data = json.load(f)
                with open(self.data_exp_path) as f:
                    self.exp_data = json.load(f)
                with open(self.data_img_sel_path) as f:
                    self.sel_data = json.load(f)
                with open(self.data_search_path) as f:
                    self.search_data = json.load(f)
                # with open(self.data_img_gen_path) as f:
                #     self.img_gen_data = json.load(f)
                self.rec_data = list(self.rec_data.values())
                self.exp_data = list(self.exp_data.values())
                self.sel_data = list(self.sel_data.values())
                self.search_data = list(self.search_data.values())
                # self.img_gen_data = list(self.img_gen_data.values())
                self.seqs = self.rec_data+self.exp_data+self.sel_data+self.search_data
                self.tasks = ["rec"]*len(self.rec_data)+["exp"]*len(self.exp_data)\
                +["img_sel"]*len(self.sel_data)+["search"]*len(self.search_data)
                # +["img_gen"]*len(self.img_gen_data)
            # with open(self.data_search_path) as f:
            #     self.search_data = json.load(f)
            # self.search_data = list(self.search_data.values())
            # self.seqs = self.search_data
            # self.tasks = ["search"]*len(self.search_data)
            
        else:
            #!!!
            self.data_path = os.path.join(self.folder_path, f"test_users.json")
            self.data_search_path = os.path.join(self.folder_path, f"test_users.json")
            #!!!
            if self.tasks=="rec":
                with open(self.data_path) as f:
                    self.data = json.load(f)
                self.seqs = list(self.data.values())
            elif self.tasks=="exp":
                with open(self.data_exp_path) as f:
                    self.data = json.load(f)
                self.seqs = list(self.data.values())
                # self.seqs = self.seqs[:int(0.1*len(self.seqs))]
            elif self.tasks=="img_sel":
                with open(self.data_img_sel_path) as f:
                    self.data = json.load(f)
                self.seqs = list(self.data.values())
                # self.seqs = self.seqs[:int(0.1*len(self.seqs))]
            elif self.tasks=="search":
                with open(self.data_search_path) as f:
                    self.data = json.load(f)
                self.seqs = list(self.data.values())
            elif self.tasks=="img_gen":
                # pre!!!
                # with open(self.meta_path) as f:
                #     self.data = json.load(f)
                with open(self.retrieval_data_path) as f:
                    self.retrieval_data = json.load(f)
                with open(self.data_path) as f:
                    self.data = json.load(f)
                with open(self.img_id_path) as f:
                    self.img_id2semantic = json.load(f)
                # ids = np.random.choice(len(self.retrieval_data), 16, replace=False)
                # self.seqs = [self.retrieval_data[_] for _ in ids]
                self.seqs = list(self.retrieval_data)
                # self.seqs = list(self.data.values())[:16]
            # subset
            # self.seqs = self.seqs[:int(0.1*len(self.seqs))]
            self.tasks = [self.tasks]*len(self.seqs)
            self.keys = list(self.data.keys())
        if self.subset=="all":
            self.all_items = set(range(22738))
        if self.subset=="netflix":
            self.all_items = set(range(1870))
        if self.subset=="hm":
            self.all_items = set(range(14901))
        # self.all_items = set(range(12094))
        
        # self.tasks = "rec"
        
        # self.selected_col_ids = [
        #         int(col_id) for col_id in args.selected_col_ids.split(",")
        #     ]
        # self.dtypes = [str for col_id in self.selected_col_ids]

        # self.dataset = dataset
        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])
        self.rank = args.rank


    def set_epoch(self, epoch, **unused):
        self.epoch = epoch
        
    
    def extract_meta(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        category = "Unknown" if sample["category"]=="" else sample["category"]
        category = " ".join(category.split()[:max_length])
        brand = "Unknown" if sample["brand"]=="" else sample["brand"]
        brand = " ".join(brand.split()[:max_length])
        title = "Unknown" if sample["title"]=="" else sample["title"]
        title = " ".join(title.split()[:max_length])
        # description = "Unknown" if sample["description"]=="" else sample["description"]
        price = "Unknown" if sample["price"]=="" else sample["price"]
        text = f"Category {category} Price {price} Brand {brand} Title {title}"
        return text
    
    def extract_meta_gen(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        category = "Unknown" if sample["category"]=="" else sample["category"]
        category = " ".join(category.split()[:max_length])
        brand = "Unknown" if sample["brand"]=="" else sample["brand"]
        brand = " ".join(brand.split()[:max_length])
        title = "Unknown" if sample["title"]=="" else sample["title"]
        title = " ".join(title.split()[:max_length])
        # description = "Unknown" if sample["description"]=="" else sample["description"]
        img_id = self.img_id2semantic[str(index)]
        img_id = [f"img_{id}," for i, id in enumerate(img_id)]
        img_id = "".join(img_id)
        text = f"Title {title} ID {img_id}"
        return text
    
    def extract_meta_netflix(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        year = sample[0]
        title = sample[1]
        title = " ".join(title.split()[:max_length])
        text = f"Title {title} Release Date {year}"
        return text
    
    def extract_meta_hm(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        prod_name, appearance, color, section, describe =  sample[0], sample[1], sample[2], sample[3], sample[4]
        prod_name = " ".join(prod_name.split()[:max_length])
        appearance = " ".join(appearance.split()[:max_length])
        color = " ".join(color.split()[:max_length])
        section = " ".join(section.split()[:max_length])
        # print(prod_name, appearance, color, section, describe)  Description {describe}
        describe = " ".join(describe.split()[:max_length])
        # text = f"Name {prod_name} Appearance {appearance} Color {color} Section {section} Description {describe}"
        text = f"Name {prod_name} Appearance {appearance} Color {color} Section {section}"
        # text = f"Name {prod_name}"
        # text = f"Name {prod_name} Section {section}"
        return text

    def process_train_rec_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # start = -min(20, len_seq)
        # lower = min(start+5,-1)
        # end = np.random.choice(list(range(lower,0)),1)[0]
        
        # if len_seq<5:
        #     end = -1
        #     start = -len_seq
        # else:
        #     end = np.random.choice(list(range(-(len_seq-4),0)),1)[0]
        #     maxi = max(-len_seq, end-20)
        #     start = np.random.choice(list(range(maxi,end-3)),1)[0]
            
        # start=-20
        # end=-1
        
        # eqaul sequence length
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            # if not self.use_semantic:
            #     id_ = f"item_{item}"
            # else:
            #     id_ = self.id2semantic[str(item)].split(",")
            #     # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            #     id_ = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(id_)]
            #     id_ = " ".join(id_)
            if not self.use_semantic:
                item_seq = f"<image> {meta_item} <answer> item_{item} <|endofchunk|> "
                # item_seq = f"<image> {meta_item} <answer> item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} <answer> {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        if not self.use_semantic:
            # naive id
            input_seq = input_seq+f"What is the next item recommended to the user? <answer> item_{seq[end]}"
            # input_seq = input_seq+f"What is the next item recommended to the user? <answer> item_domain_{seq[end]}"
        else:
            # semantic id
            item = seq[end]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
            input_seq = input_seq+f"What is the next item recommended to the user? <answer> {semantic_id}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(2.0)
            }
        }

        return example
    
    def process_eval_rec_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        if self.subset=="hm":
            test_len = 20
        else:
            test_len = 5
        for item in seq[-test_len:-1]:
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            # p = np.random.random()
            # if p<0.3:
            #     image_item = Image.open(os.path.join(self.img_folder, f"0.jpg")).convert("RGB")
            # else:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            img_seq.append(self.patch_resize_transform(image_item))
            # semantic_id = f"item_{item}"
            # item_seq = f"<image> {meta_item} <|endofchunk|> "
            if not self.use_semantic:
                item_seq = f"<image> {meta_item} item_{item} <|endofchunk|> "
                # item_seq = f"<image> {meta_item} item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        input_seq = input_seq+f"What is the next item recommended to the user? <answer>"
        # naive id
        if not self.use_semantic:
            semantic_id = f"item_{seq[-1]}"
            # semantic_id = f"item_domain_{seq[-1]}"
        else:
            # semantic id
            item = seq[-1]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
        
        input_len = len(input_seq.split(" "))
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        
        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": semantic_id,
            }
        }

        return example
    
    def process_pre_train_img_gen_pair(self, index):
        item = self.keys[index]
        img_seq = []
        input_seq = ""        
        image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}" for i, id in enumerate(img_id)]
        img_id = " ".join(img_id)
        category = self.meta_data[str(item)]["category"]
        category = " ".join(category.split()[4:])
        query = self.meta_data[str(item)]["title"]
        query = " ".join(query.split()[:30])
        # input_seq = input_seq+f"Query: {query} What is the generated image ID to the query based on the history? <answer> {img_id}"
        input_seq = f"Query: {query}. What is the generated image ID to the query? <answer> {img_id}"
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }

        return example
    
    def process_pre_eval_img_gen_pair(self, index):
        item = self.keys[index]
        img_seq = []
        input_seq = ""
        image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}" for i, id in enumerate(img_id)]
        img_id = " ".join(img_id)
        category = self.meta_data[str(item)]["category"]
        category = " ".join(category.split()[4:])
        query = self.meta_data[str(item)]["title"]
        query = " ".join(query.split()[:30])
        input_seq = f"Query: {query}. What is the generated Image ID to the query? <answer>"
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": img_id,
                "items": int(item)
            }
        }

        return example
    
    def process_retrieve_train_img_gen_pair(self, index):
        seq = self.retrieval_data[index]
        img_seq = []
        input_seq = ""
        end = -1
        start = end-self.history_len
        # start=-5
        # end=-1
        
        # normal sequence
        
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # ID: {img_id} 
            meta_item = self.extract_meta_gen(item)
            item_seq = f"<image> {meta_item} <|endofchunk|> "
            # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        
        item=seq[end]
        # predict
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}," for i, id in enumerate(img_id)]
        img_id = "".join(img_id)
        # query = self.meta_data[str(item)]["category"]
        query = self.meta_data[str(item)]["keywords"]
        query = " ".join(query.split()[:30])
        input_seq = input_seq+f"Query: {query} What is the generated image ID to the query based on the history? <answer> {img_id}"
        # input_seq = f"Query: {query}. What is the generated image ID to the query? <answer> {img_id}"
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        
        input_len = len(input_seq.split(" "))

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }
        return example
    
    def process_retrieve_eval_img_gen_pair(self, index):
        seq = self.retrieval_data[index]
        img_seq = []
        input_seq = ""
        end = -1
        start = end-self.history_len
        
        # ori seq
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # ID: {img_id} 
            meta_item = self.extract_meta_gen(item)
            item_seq = f"<image> {meta_item} <|endofchunk|> "
            # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
            input_seq = input_seq+item_seq
            
        item=seq[end]
        # predict
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}," for i, id in enumerate(img_id)]
        img_id = "".join(img_id)
        # query = self.meta_data[str(item)]["category"]
        query = self.meta_data[str(item)]["keywords"]
        query = " ".join(query.split()[:30])
        input_seq = input_seq+f"Query: {query} What is the generated Image ID to the query based on the history? <answer>"
        # input_seq = f"Query: {query}. What is the generated Image ID to the query? <answer>"
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": img_id,
                "items": item
            }
        }

        return example
    
    def process_train_img_gen_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        # start=-5
        # end=-1
        
        # normal sequence
        
        # for item in seq[start:end]:
        #     image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        #     img_seq.append(self.patch_resize_transform(image_item))
        #     # ID: {img_id} 
        #     meta_item = self.extract_meta_gen(item)
        #     item_seq = f"<image> {meta_item} <|endofchunk|> "
        #     # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
        #     input_seq = input_seq+item_seq
        
        item=seq[end]
        # retrieval
        retrieval = self.meta_data[str(item)]["retrieval"][0]
        image_item = Image.open(os.path.join(self.img_folder, f"{retrieval}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        meta_item = self.extract_meta_gen(retrieval)
        item_seq = f"<image> {meta_item} <|endofchunk|> "
        input_seq = input_seq+item_seq
        # predict
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}," for i, id in enumerate(img_id)]
        img_id = "".join(img_id)
        # query = self.meta_data[str(item)]["category"]
        query = self.meta_data[str(item)]["keywords"]
        query = " ".join(query.split()[:30])
        input_seq = input_seq+f"Query: {query} What is the generated image ID to the query based on the history? <answer> {img_id}"
        # input_seq = f"Query: {query}. What is the generated image ID to the query? <answer> {img_id}"
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }
        return example
    
    def process_eval_img_gen_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        
        # ori seq
        # for item in seq[-self.history_len-1:-1]:
        #     # p = np.random.random()
        #     # if p<1.1:
        #     #     item_seq = f"{item} <|endofchunk|> "
        #     # else:
        #     image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        #     img_seq.append(self.patch_resize_transform(image_item))
        #     meta_item = self.extract_meta_gen(item)
        #     item_seq = f"<image> {meta_item} <|endofchunk|> "
        #     # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
        #     # item_seq = f"<image> item_{item} <|endofchunk|> "
        #     input_seq = input_seq+item_seq
            
        item=seq[-1]
        # retrieval
        retrieval = self.meta_data[str(item)]["retrieval"][0]
        image_item = Image.open(os.path.join(self.img_folder, f"{retrieval}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        meta_item = self.extract_meta_gen(retrieval)
        item_seq = f"<image> {meta_item} <|endofchunk|> "
        input_seq = input_seq+item_seq
        # predict
        img_id = self.img_id2semantic[str(item)]
        img_id = [f"img_{id}" for i, id in enumerate(img_id)]
        img_id = " ".join(img_id)
        # query = self.meta_data[str(item)]["category"]
        query = self.meta_data[str(item)]["keywords"]
        query = " ".join(query.split()[:30])
        input_seq = input_seq+f"Query: {query} What is the generated Image ID to the query based on the history? <answer>"
        # input_seq = f"Query: {query}. What is the generated Image ID to the query? <answer>"
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": img_id,
                "items": item
            }
        }

        return example
    
    def process_train_search_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # if len_seq>5:
        #     start = np.random.choice(range(-len_seq, -4),1)[0]
        #     end = start+4
        # else:
        #     start=-5
        #     end=-1
        # start=-5
        # end=-1
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # category = self.meta_data[str(item)]["category"]
            meta_item = self.extract_meta(item)
            if not self.use_semantic:
                item_seq = f"<image> {meta_item} <answer> item_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = " ".join(semantic_id)
                item_seq = f"<image> {meta_item} <answer> {semantic_id} <|endofchunk|> "
            # item_seq = f"<image> {meta_item} <|endofchunk|> "
            # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        
        item=seq[end]
        if self.subset=="cloth":
            query = self.meta_data[str(item)]["keywords"]
        else:
            query = self.meta_data[str(item)]["category"]
        # input_seq = input_seq+f"Query: {query}. What is the next item recommended to the user? <answer> item_{item}"
        # input_seq = input_seq+f"Query: {query} What is the related item ID to the query based on the history? <answer> item_{item}"
        if not self.use_semantic:
            # naive id
            input_seq = input_seq+f"Query: {query} What is the related item ID to the query based on the history? <answer> item_{item}"
        else:
            # semantic id
            item = seq[end]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = " ".join(semantic_id)
            input_seq = input_seq+f"Query: {query} What is the related item ID to the query based on the history? <answer> {semantic_id}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }

        return example
    
    def process_eval_search_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        for item in seq[-5:-1]:
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            if not self.use_semantic:
                item_seq = f"<image> {meta_item} item_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = " ".join(semantic_id)
                item_seq = f"<image> {meta_item} {semantic_id} <|endofchunk|> "
            # item_seq = f"<image> {meta_item} <|endofchunk|> "
            # item_seq = f"<image> Query: {category} ID: item_{item} <|endofchunk|> "
            # item_seq = f"<image> item_{item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        item=seq[-1]
        if self.subset=="cloth":
            query = self.meta_data[str(item)]["keywords"]
        else:
            query = self.meta_data[str(item)]["category"]
        input_seq = input_seq+f"Query: {query} What is the related item ID to the query based on the history? <answer>"
        if not self.use_semantic:
            semantic_id = f"item_{seq[-1]}"
        else:
            # semantic id
            item = seq[-1]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = " ".join(semantic_id)
        # input_seq = input_seq+f"What is the next item recommended to the user? <answer>"
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": semantic_id,
            }
        }

        return example
    
    def process_train_img_sel_pair(self, index):
        full_seq = self.seqs[index]
        img_seq = []
        input_seq = "User history: "
        # start = np.random.choice(list(range(-15,-5)),1)[0]
        num_items = 3
        start = -(self.history_len-num_items+1)
        cur_items = []
        for full_item in full_seq[start:-1]:
            item=full_item[0]
            cur_items.append(item)
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            item_seq = f"<image> {meta_item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        input_seq = input_seq+"Select from: "
        item_set = full_seq[-1][-2]
        # randomly choose negs
        for index in full_seq[-1][-1]:
            cur_items.append(item_set[index])
        gt_index = full_seq[-1][-1]
        gt_items = [item_set[index] for index in gt_index]
        len_gt = len(gt_items)
        labels = np.random.choice(list(range(num_items)), len_gt, replace=False)
        neg_index = list(set(range(num_items))-set(labels))
        negs = np.random.choice(list(self.all_items-set(cur_items)), num_items-len_gt,replace=False)
        item_set = [0]*num_items
        for i, item in enumerate(gt_items):
            item_set[labels[i]] = item
        for i, item in enumerate(negs):
            item_set[neg_index[i]] = item
        for i, item in enumerate(item_set):
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            # item_seq = f"<image> {item} <|endofchunk|> "
            item_seq = f"<image> Selection s_{i} {meta_item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        input_seq = input_seq+f"Can you select the suitable item from above for the user? <answer> "
        # labels = [item_set[index] for index in full_seq[-1][-1]]
        # labels = full_seq[-1][-1]
        for label in labels:
            # input_seq = input_seq+f"{label} "
            input_seq = input_seq+f"s_{label} "
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }

        return example
    
    def process_eval_img_sel_pair(self, index):
        full_seq = self.seqs[index]
        img_seq = []
        input_seq = "User history: "
        for full_item in full_seq[-5:-1]:
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            item = full_item[0]
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            item_seq = f"<image> {meta_item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        input_seq = input_seq+"Select from: "
        item_set = full_seq[-1][-2]
        for i, item in enumerate(item_set):
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            item_seq = f"<image> Selection s_{i} {meta_item} <|endofchunk|> "
            # item_seq = f"<image> {item} <|endofchunk|> "
            input_seq = input_seq+item_seq
        input_seq = input_seq+f"Can you select the suitable item from above for the user? <answer>"
        # labels = [item_set[index] for index in full_seq[-1][-1]]
        labels = full_seq[-1][-1]
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ids": torch.tensor(labels),
            }
        }

        return example
    
    def process_train_rate_exp_pair(self, index):
        full_seq = self.seqs[index]
        img_seq = []
        input_seq = ""
        # start = np.random.choice(list(range(-10,-5)),1)[0]
        # end = np.random.choice(list(range(start+5,0)),1)[0]
        len_seq = len(full_seq)
        # if len_seq>5:
        #     start = np.random.choice(range(-len_seq, -4),1)[0]
        #     end = start+4
        # else:
        #     start=-5
        #     end=-1
        start = np.random.choice(list(range(0,len_seq-self.history_len+1)), 1)[0]
        end = start+self.history_len-1
        max_length=30
        for full_item in full_seq[start:end]:
            item = full_item[0]
            exp = " ".join(full_item[1].split()[:max_length])
            # exp = full_item[1]
            rate = int(full_item[2])
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            item_seq = f"<image> {meta_item} <answer> rate_{rate} {exp} <|endofchunk|> "
            input_seq = input_seq+item_seq
        # handle the ground truth for prediction
        full_item = full_seq[end]
        item = full_item[0]
        exp = " ".join(full_item[1].split()[:max_length])
        rate = int(full_item[2])
        image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        meta_item = self.extract_meta(item)
        input_seq = input_seq+f"<image> {meta_item} What is the rating and explanation for the item? <answer> rate_{rate} {exp}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(1.0)
            }
        }

        return example
    
    def process_eval_rate_exp_pair(self, index):
        full_seq = self.seqs[index]
        img_seq = []
        input_seq = ""
            
        for full_item in full_seq[-5:-1]:
            item = full_item[0]
            exp = full_item[1]
            # exp = " ".join(full_item[1].split()[:max_length])
            rate = int(full_item[2])
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            meta_item = self.extract_meta(item)
            item_seq = f"<image> {meta_item} <answer> rate_{rate} {exp} <|endofchunk|> "
            input_seq = input_seq+item_seq
        # handle the ground truth for prediction
        full_item = full_seq[-1]
        item = full_item[0]
        exp = full_item[1]
        # exp = " ".join(full_item[1].split()[:max_length])
        rate = int(full_item[2])
        image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        meta_item = self.extract_meta(item)
        input_seq = input_seq+f"<image> {meta_item} What is the rating and explanation for the item? <answer>"
        
        input_len = len(input_seq.split(" "))
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len
            },
            "net_output":{
                "output_ratings": [rate],
                "output_exps": [exp]
            }
        }

        return example

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        self.task = self.tasks[index]
        if self.task=="rec":
            # if self.split=="train":
            #     pair_samples = self.process_train_semantic_rec_pair(index)
            # else:
            #     pair_samples = self.process_eval_semantic_rec_pair(index)
            if self.split=="train":
                pair_samples = self.process_train_rec_pair(index)
            else:
                pair_samples = self.process_eval_rec_pair(index)
        elif self.task=="exp":
            if self.split=="train":
                pair_samples = self.process_train_rate_exp_pair(index)
            else:
                pair_samples = self.process_eval_rate_exp_pair(index)
        elif self.task=="img_sel":
            if self.split=="train":
                pair_samples = self.process_train_img_sel_pair(index)
            else:
                pair_samples = self.process_eval_img_sel_pair(index)
        elif self.task=="search":
            if self.split=="train":
                pair_samples = self.process_train_search_pair(index)
            else:
                pair_samples = self.process_eval_search_pair(index)
        # elif self.task=="img_gen":
        #     if self.split=="train":
        #         pair_samples = self.process_train_img_gen_pair(index)
        #     else:
        #         pair_samples = self.process_eval_img_gen_pair(index)
        elif self.task=="img_gen":
            if self.split=="train":
                pair_samples = self.process_retrieve_train_img_gen_pair(index)
            else:
                pair_samples = self.process_retrieve_eval_img_gen_pair(index)
        else:
            raise KeyError("Not Supported Task")
        # if dataset is not supported
        if pair_samples is None:
            return self.__getitem__(index + 1)
        return pair_samples
    
    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1
    

    # def collate(self, samples):
    #     """Merge samples of different tasks to form two mini-batches.
    #     Args:
    #         samples (List[Tuple]): samples to collate
    #     Returns:
    #         Tuple[dict]: two mini-batch containing the data of different tasks
    #     """

    #     samples_v1 = []  # containing image-text pairs
    #     for sample_tuple in samples:
    #         samples_v1.append(sample_tuple[0])

    #     res_v1 = collate_fn(
    #         samples_v1,
    #         pad_idx=self.tokenizer.pad_token_id,
    #         eos_idx=self.tokenizer.eos_token_id,
    #     )
    #     return res_v1

