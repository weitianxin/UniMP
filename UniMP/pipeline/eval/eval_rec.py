import argparse
import glob
import os
import random
import json
import numpy as np
import torch
import torch.nn
import wandb
from pipeline.train.data import get_data
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.eval.rec_metrics import ndcg_at_k, hit_at_k, mrr_at_k
from pipeline.eval.utils import NumpyEncoder

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


def eval_model_rec(
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
    hr, ndcg, mrr=[], [], []
    hr_5, ndcg_5, mrr_5=[], [], []
    hr_3, ndcg_3, mrr_3=[], [], []
    K=10
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
        input_length = batch_multi_instruct["net_input"]["input_len"][0]
        output_ids = batch_multi_instruct["net_output"]["output_ids"][0]
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
                    max_new_tokens=50,
                    eos_token_id=args.tokenizer.eos_token_id,
                    pad_token_id=args.tokenizer.eos_token_id
                )
            texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        texts = [text.split("</s>")[0] for text in texts]
        # texts = list(set(["".join(text.split()[input_length:]).strip() for text in texts]))
        texts = [(text.split("?")[-1]).strip() for text in texts]
        # naive id
        # target = str(output_ids.item())
        # item_target = f"item_{target}"
        
        # both naive id and semantic id
        item_target = output_ids
        
        if num_steps==0 and args.rank==0:
            print(texts)
            print(item_target)
        gen_ids = [text==item_target for text in texts]
        r = np.array([0]*10)
        r_ = np.array(gen_ids,dtype=int)
        r[:len(r_)] = r_
        
        user_mrr_3 = mrr_at_k(r, 3)
        user_hr_3 = hit_at_k(r, 3)
        user_ndcg_3 = ndcg_at_k(r, 3, 1)
        hr_3.append(user_hr_3)
        ndcg_3.append(user_ndcg_3)
        mrr_3.append(user_mrr_3)
        
        user_mrr_5 = mrr_at_k(r, 5)
        user_hr_5 = hit_at_k(r, 5)
        user_ndcg_5 = ndcg_at_k(r, 5, 1)
        hr_5.append(user_hr_5)
        ndcg_5.append(user_ndcg_5)
        mrr_5.append(user_mrr_5)
        
        user_mrr = mrr_at_k(r, K)
        user_hr = hit_at_k(r, K)
        # len_gt
        user_ndcg = ndcg_at_k(r, K, 1)
        hr.append(user_hr)
        ndcg.append(user_ndcg)
        mrr.append(user_mrr)
        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()
    rec_metrics = np.array([hr_3, ndcg_3, mrr_3, hr_5, ndcg_5, mrr_5, hr, ndcg, mrr])
    hr_mean_3, ndcg_mean_3, mrr_mean_3 = np.mean(hr_3), np.mean(ndcg_3), np.mean(mrr_3)
    hr_mean_5, ndcg_mean_5, mrr_mean_5 = np.mean(hr_5), np.mean(ndcg_5), np.mean(mrr_5)
    hr_mean, ndcg_mean, mrr_mean = np.mean(hr), np.mean(ndcg), np.mean(mrr)
    with open(f"results/{args.run_name}_rec_epoch_{epoch}_rank_{args.rank}.txt","w") as f:
        json.dump(rec_metrics, f, cls=NumpyEncoder)
    # accelerator.wait_for_everyone()
    # hr_all_3, ndcg_all_3, mrr_all_3 = accelerator.gather((torch.tensor(hr_3).to(args.device), torch.tensor(ndcg_3).to(args.device), torch.tensor(mrr_3).to(args.device)))
    # accelerator.wait_for_everyone()
    # hr_all_5, ndcg_all_5, mrr_all_5 = accelerator.gather((torch.tensor(hr_5).to(args.device), torch.tensor(ndcg_5).to(args.device), torch.tensor(mrr_5).to(args.device)))
    # accelerator.wait_for_everyone()
    # hr_all, ndcg_all, mrr_all = accelerator.gather((torch.tensor(hr).to(args.device), torch.tensor(ndcg).to(args.device), torch.tensor(mrr).to(args.device)))
    # accelerator.wait_for_everyone()
    # hr_mean_3, ndcg_mean_3, mrr_mean_3 = torch.mean(hr_all_3), torch.mean(ndcg_all_3), torch.mean(mrr_all_3)
    # hr_mean_5, ndcg_mean_5, mrr_mean_5 = torch.mean(hr_all_5), torch.mean(ndcg_all_5), torch.mean(mrr_all_5)
    # hr_mean, ndcg_mean, mrr_mean = torch.mean(hr_all), torch.mean(ndcg_all), torch.mean(mrr_all)
    if args.rank == 0 and args.report_to_wandb:
        wandb.log(
            {
                "hr@10": hr_mean,
                "ndcg@10": ndcg_mean,
                "mrr@10": mrr_mean,
                "hr@5": hr_mean_5,
                "ndcg@5": ndcg_mean_5,
                "mrr@5": mrr_mean_5,
                "hr@3": hr_mean_3,
                "ndcg@3": ndcg_mean_3,
                "mrr@3": mrr_mean_3,
            },
            commit=True,
        )
    if args.rank==0:
        # with open("results.txt","a+") as f:
        #     f.write(f"Epoch {epoch}\n")
        print("epoch: ", epoch)
        print(f"HR@3: {hr_mean_3} NDCG@3: {ndcg_mean_3} MRR@3: {mrr_mean_3} HR@5: {hr_mean_5} NDCG@5: {ndcg_mean_5} MRR@5: {mrr_mean_5} HR@10: {hr_mean} NDCG@10: {ndcg_mean} MRR@10: {mrr_mean}")
    
    
