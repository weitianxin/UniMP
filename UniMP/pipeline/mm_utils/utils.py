def process_train_semantic_rec_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # start = -min(20, len_seq)
        # lower = min(start+5,-1)
        # end = np.random.choice(list(range(lower,0)),1)[0]
        if len_seq<6:
            end = -1
            start = -len_seq
        else:
            end = np.random.choice(list(range(-(len_seq-4),0)),1)[0]
            maxi = max(-len_seq, end-20)
            start = np.random.choice(list(range(maxi,end-3)),1)[0]
        # start=-20
        # end=-1
        for index, item in enumerate(seq[start:end]):
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            semantic_id = self.id2semantic[str(item)].split(",")
            semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            # semantic_id = [f"item_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = " ".join(semantic_id)
            item_seq = f"<image> {index} <|endofchunk|> "
            # item_seq = f"<image> {semantic_id} <|endofchunk|> "
            # item_seq = f"<image> <answer> {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        item = seq[end]
        semantic_id = self.id2semantic[str(item)].split(",")
        semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
        # semantic_id = [f"item_{id}" for i, id in enumerate(semantic_id)]
        semantic_id = " ".join(semantic_id)
        input_seq = input_seq+f"What is the next item recommended to the user? <answer> {semantic_id}"
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
                "weights": torch.tensor(2.0)
            }
        }

        return example
    
def process_eval_semantic_rec_pair(self, index):
    full_seq = self.seqs[index]
    seq = [item[0] for item in full_seq]
    img_seq = []
    input_seq = ""
    for index, item in enumerate(seq[-20:-1]):
        # p = np.random.random()
        # if p<1.1:
        #     item_seq = f"{item} <|endofchunk|> "
        # else:
        image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
        img_seq.append(self.patch_resize_transform(image_item))
        semantic_id = self.id2semantic[str(item)].split(",")
        semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
        # semantic_id = [f"item_{id}" for i, id in enumerate(semantic_id)]
        semantic_id = " ".join(semantic_id)
        item_seq = f"<image> {index} <|endofchunk|> "
        # item_seq = f"<image> <answer> {semantic_id} <|endofchunk|>
        # item_seq = f"<image> {semantic_id} <|endofchunk|> "
        # item_seq = f"<image> <answer> {semantic_id} <|endofchunk|> "
        input_seq = input_seq+item_seq
    item = seq[-1]
    semantic_id = self.id2semantic[str(item)].split(",")
    semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
    # semantic_id = [f"item_{id}" for i, id in enumerate(semantic_id)]
    semantic_id = " ".join(semantic_id)
    input_seq = input_seq+f"What is the next item recommended to the user? <answer>"
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