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

from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
import time

def eval_model_img_sel(
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
    recall_list, precision_list, f1_list = [], [], []
    K=1
    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
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
        images = (
            batch_multi_instruct["net_input"]["patch_images"].unsqueeze(2)
        )
        input_ids = batch_multi_instruct["net_input"]["input_ids"]
        attention_mask = batch_multi_instruct["net_input"]["attention_masks"]
        # just for seq rec
        output_ids = batch_multi_instruct["net_output"]["output_ids"][0].cpu().detach().numpy()
        id_length = input_ids[0].shape[0]
        with torch.no_grad():
            with autocast():
                generated_texts = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    num_beams=2,
                    num_return_sequences=K,
                    early_stopping=True,
                    max_new_tokens=40,
                    no_repeat_ngram_size=0,
                    eos_token_id=args.tokenizer.eos_token_id,
                    pad_token_id=args.tokenizer.eos_token_id
                )
            texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        gen_ids = set(texts[0].split("?")[-1].strip().split())
        gts = [f"s_{output_id}" for output_id in output_ids]
        # gts = [str(int(id))for id in output_ids]
        len_gt = len(gts)
        len_gen = len(gen_ids)
        
        r = np.sum([1 if gen_id in gts else 0 for gen_id in gen_ids],dtype=np.float64)
        
        recall = r/len_gt
        if len_gen==0:
            precision = 0
        else:
            precision = r/len_gen
        if args.rank==0 and num_steps==0:
            print(gts, gen_ids)
        if precision>0 or recall>0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
                
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()
    recall_all, precision_all, f1_all = accelerator.gather((torch.tensor(recall_list).to(device_id), 
                                                                torch.tensor(precision_list).to(device_id), 
                                                                torch.tensor(f1_list).to(device_id)))
    recall_mean, precision_mean, f1_mean = torch.mean(recall_all), torch.mean(precision_all), torch.mean(f1_all)
    if args.rank == 0 and args.report_to_wandb:
        
        wandb.log(
            {
                "recall": recall_mean,
                "precision": precision_mean,
                "f1": f1_mean
            },
            commit=True,
        )
    if args.rank==0:
        print(f"recall: {recall_mean},precision: {precision_mean},f1: {f1_mean}")
