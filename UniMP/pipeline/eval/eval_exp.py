import argparse
import glob
import os
import random, json

import numpy as np
import torch
import torch.nn
import wandb
from pipeline.train.data import get_data
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.eval.rec_metrics import ndcg_at_k, hit_at_k
from utils import convert_to_json
from metric.evaluator import get_evaluator
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    CLIPImageProcessor,
)
import datasets, evaluate
from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
import time

def eval_model_exp(
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
    eval_embed=False
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
    mae = []
    rmse = []
    K=1
    task = 'summarization'
    gen_exps_all, real_exps_all = [], []
    metric_1 = evaluate.load('bleu')
    metric_2 = evaluate.load('rouge')
    metric_4 = evaluate.load("meteor")
    if eval_embed:
        metric_3 = evaluate.load('bertscore')
    # loop througe dataloader
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
        id_length = input_ids[0].shape[0]
        with torch.no_grad():
            with autocast():
                generated_texts = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    num_beams=5,
                    num_return_sequences=K,
                    early_stopping=True,
                    max_new_tokens=256,
                    eos_token_id=args.tokenizer.eos_token_id,
                    pad_token_id=args.tokenizer.eos_token_id
                    # no_repeat_ngram_size=0
                )
            texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        # texts = list(set(["".join(text.split()[input_length:]).strip() for text in texts]))
        texts = list([text.split("?")[-1].strip().split() for text in texts])
        gen_rates = []
        for text in texts:
            try:
                gen_rates.append(float(text[0].split("_")[-1]))
            except:
                gen_rates.append(3.0)
        gen_rates = np.array(gen_rates,dtype=np.float64)
        gen_exps = [" ".join(text[1:]) for text in texts]
        gen_exps = ["Empty" if exp=="" else exp for exp in gen_exps]
        real_rates = batch_multi_instruct["net_output"]["output_ratings"][0].cpu().detach().numpy()
        real_exps = batch_multi_instruct["net_output"]["output_exps"]
        mae.extend(np.abs(gen_rates-real_rates))
        rmse.extend(np.square(gen_rates-real_rates))
        metric_1.add_batch(predictions=gen_exps, references=real_exps)
        metric_2.add_batch(predictions=gen_exps, references=real_exps)
        metric_4.add_batch(predictions=gen_exps, references=real_exps)
        if eval_embed:
            metric_3.add_batch(predictions=gen_exps, references=real_exps)
            gen_exps_all.extend(gen_exps)
            real_exps_all.extend(real_exps)
        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()
    mae_all = accelerator.gather_for_metrics(torch.tensor(mae).to(device_id))
    rmse_all = accelerator.gather_for_metrics(torch.tensor(rmse).to(device_id))
    bleu = metric_1.compute()["precisions"][0]
    rouge = metric_2.compute()
    rouge1, rouge2, rougeL = rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]
    meteor = metric_4.compute()["meteor"]
    bleu_all, rouge1_all, rouge2_all, rougeL_all, meteor_all = accelerator.gather((torch.tensor(bleu).to(device_id),torch.tensor(rouge1).to(device_id), 
                                                                                   torch.tensor(rouge2).to(device_id), 
                                                                                    torch.tensor(rougeL).to(device_id),
                                                                                    torch.tensor(meteor).to(device_id)))
    
    if eval_embed:
        bertscore = metric_3.compute(lang="en")["f1"]
        bertscore_all = accelerator.gather_for_metrics(torch.tensor(bertscore).to(device_id))
        if not os.path.exists("save_gen"):
            os.mkdir("save_gen")
        with open(f"save_gen/gen_exps_{args.rank}.json","w") as f:
            json.dump(gen_exps_all ,f)
        with open(f"save_gen/real_exps_{args.rank}.json","w") as f:
            json.dump(real_exps_all ,f)
    mae_all = torch.mean(mae_all)
    rmse_all  = torch.sqrt(torch.mean(rmse_all))
    bleu_all = torch.mean(bleu_all)
    rouge1_all = torch.mean(rouge1_all)
    rouge2_all = torch.mean(rouge2_all)
    rougeL_all = torch.mean(rougeL_all)
    meteor_all = torch.mean(meteor_all)
    output_seq = f"rmse: {rmse_all} \nmae: {mae_all} \nbleu: {bleu_all} \nrouge1 {rouge1_all} \nrouge2 {rouge2_all} \nrougeL {rougeL_all} \nmeteor {meteor_all}\n"
    if eval_embed:
        bertscore_all = torch.mean(bertscore_all)
        output_seq += f"bertscore {bertscore_all}\n"
    if args.rank==0:
        print(output_seq)
        with open("results_exp.txt","a+") as f:
            f.write(output_seq+"\n")
    if args.rank == 0 and args.report_to_wandb:
        if eval_embed:
            wandb.log(
                {
                    "mae": mae_all,
                    "bleu": bleu_all,
                    "rouge1":rouge1_all,
                    "rouge2":rouge2_all,
                    "rougeL":rougeL_all,
                    "bertscore":bertscore_all,
                    "meteor": meteor_all,
                    "rmse": rmse_all
                },
                commit=True,
            )
        else:
            wandb.log(
                {
                    "mae": mae_all,
                    "bleu": bleu_all,
                    "rouge1":rouge1_all,
                    "rouge2":rouge2_all,
                    "rougeL":rougeL_all,
                    "meteor": meteor_all,
                    "rmse": rmse_all
                },
                commit=True,
            )
    
        
