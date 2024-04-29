import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn
import wandb
from pipeline.train.data import get_data
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.eval.rec_metrics import ndcg_at_k, hit_at_k
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    CLIPImageProcessor,
)
import json
from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
import time

def eval_model_img_gen(
    args,
    model,
    epoch,
    multi_instruct_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator,
    wandb,
):
    # num_batches_per_epoch = multi_instruct_loader.num_batches
    num_batches_per_epoch = len(multi_instruct_loader)
    total_training_steps = num_batches_per_epoch

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.eval()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    end = time.time()
    hr, ndcg=[], []
    K=1
    texts_dict = {} 
    #do_sample=True,
    # top_k=50,
    # temperature=1.
    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0
    ):

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MULTI_INSTRUCT FORWARD PASS ####

        # images = (
        #     batch_multi_instruct["net_input"]["patch_images"]
        #     .to(device_id, dtype=cast_dtype, non_blocking=True)
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        # )
        
        # images = (
        #     batch_multi_instruct["net_input"]["patch_images"].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2)
        # )
        # input_ids = batch_multi_instruct["net_input"]["input_ids"].to(
        #     device_id, dtype=cast_dtype, non_blocking=True
        # )
        # attention_mask = batch_multi_instruct["net_input"]["attention_masks"].to(
        #     device_id, dtype=cast_dtype, non_blocking=True
        # )
        
        # do_sample=True,
        # top_k=100,
        # temperature=1.
        images = (
            batch_multi_instruct["net_input"]["patch_images"].unsqueeze(2)
        )
        input_ids = batch_multi_instruct["net_input"]["input_ids"]
        attention_mask = batch_multi_instruct["net_input"]["attention_masks"]
        # just for seq rec
        input_length = batch_multi_instruct["net_input"]["input_len"][0]
        output_ids = batch_multi_instruct["net_output"]["output_ids"][0]
        items = batch_multi_instruct["net_output"]["items"][0]
        id_length = input_ids[0].shape[0]
        with torch.no_grad():
            with autocast():
                generated_texts = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    num_beams=K,
                    num_return_sequences=K,
                    early_stopping=True,
                    max_new_tokens=600,
                    eos_token_id=args.tokenizer.eos_token_id,
                    pad_token_id=args.tokenizer.eos_token_id
                )
            texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        texts = [text.split("</s>")[0] for text in texts]
        # texts = list(set(["".join(text.split()[input_length:]).strip() for text in texts]))
        texts = [(text.split("?")[-1]).strip() for text in texts]
        texts_dict[items.item()] = texts
        
        # naive id
        # target = str(output_ids.item())
        # item_target = f"item_{target}"
        
        # semantic id
        item_target = output_ids
        if num_steps==0 and args.rank==0:
            print(texts)
            print(item_target)
        # gen_ids = [text==item_target for text in texts]
        # r = np.array([0]*10)
        # r_ = np.array(gen_ids,dtype=int)
        # r[:len(r_)] = r_
        # user_hr = hit_at_k(r, K)
        # user_ndcg = ndcg_at_k(r, K, 1)
        # hr.append(user_hr)
        # ndcg.append(user_ndcg)
        
        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()
    # hr_all, ndcg_all = accelerator.gather_for_metrics((torch.tensor(hr).to(device_id), torch.tensor(ndcg).to(device_id)))
    if not os.path.exists("save_img_gen"):
            os.mkdir("save_img_gen")
    with open(f"save_img_gen/img_gen_{args.rank}_epoch_{epoch}_name_{args.run_name}.json","w") as f:
        json.dump(texts_dict ,f)
    # if args.rank == 0 and args.report_to_wandb:
        # hr_mean, ndcg_mean = torch.mean(hr_all), torch.mean(ndcg_all)
        # wandb.log(
        #     {
        #         "hr@10": hr_mean,
        #         "ndcg@10": ndcg_mean
        #     },
        #     commit=True,
        # )
        # print(f"HR@10: {hr_mean} NDCG@10: {ndcg_mean}")
        # with open("results.txt","a+") as f:
        #     f.write(f"HR@10: {hr_mean} NDCG@10: {ndcg_mean}\n")
