""" Main training script """

import argparse
import glob
import os
import random
# from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import numpy as np
import torch
import torch.nn
import wandb
from pipeline.train.data import get_data_rec
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.eval.eval_rec import eval_model_rec
from pipeline.eval.eval_exp import eval_model_exp
from pipeline.eval.eval_img_sel import eval_model_img_sel
from pipeline.eval.eval_search import eval_model_search
from pipeline.eval.eval_img_gen import eval_model_img_gen

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from open_flamingo import Flamingo

import warnings
warnings.filterwarnings("ignore")
os.environ["NCCL_P2P_LEVEL"] = "NVL"
# os.environ["NCCL_P2P_DISABLE"] = "1"
         
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

from pipeline.mm_utils.arguments import add_data_args
from accelerate import Accelerator

import sys

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_one_epoch(
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
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    if args.task=="img_gen":
        img_tokens = ["img_789", "img_591", "img_977"]
        delete_list = []
        for token in img_tokens:
            img_token = tokenizer(token, add_special_tokens=False)["input_ids"][-1]
            delete_list.append(img_token)
    model.train()
    # alpha = 0.25
    gamma = args.gamma
    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

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
        weights = batch_multi_instruct["net_input"]["weights"]

        labels = input_ids.clone()
        # reweight = (1-alpha)*torch.ones_like(labels, dtype=torch.float)
        # only keep the loss for eos and the answer between <answer> and <endofchunk>
        for i in range(labels.shape[0]):
            answer_flag=0
            for j in range(labels.shape[1]):
                if not answer_flag:
                    if labels[i, j] == answer_token_id:
                        answer_flag=1
                    labels[i, j] = -100
                else:
                    if labels[i, j] == endofchunk_token_id:
                        answer_flag=0
                        labels[i, j] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        # for i in range(labels.shape[0]):
        #     # remove loss for any token before <answer> token
        #     label_idx = 0
        #     while (
        #         label_idx < labels.shape[1] and labels[i][label_idx] != answer_token_id
        #     ):
        #         labels[i][label_idx] = -100
        #         label_idx += 1
        labels[labels == answer_token_id] = -100
        labels[labels == media_token_id] = -100
        # if args.task=="img_gen":
        #     for delete_token in delete_list:
        #         reweight[labels == delete_token] = alpha
        #     reweight = reweight[:, 1:].contiguous()
        
        # labels.to(device_id, dtype=cast_dtype, non_blocking=True)
        with accelerator.accumulate(model):
            with autocast():
                output = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,)
                loss_multi_instruct_b = output[0]

            # divided_loss_multi_instruct = loss_multi_instruct

            #### BACKWARD PASS ####
            # loss_multi_instruct = loss_multi_instruct_b*weights[0]
            
            
            lm_logits = output["logits"]
            labels = labels.to(lm_logits.device)
            # batch_size x n_tokens
            n1, n2 = labels.shape[0], labels.shape[1]-1
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # resize
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            labels = labels.view(-1)
            # loss is zero for label index = 100
            lm_loss = loss_fct(shift_logits, labels).view(n1,n2)
            # task weight
            loss_multi_instruct = torch.unsqueeze(weights,1)*lm_loss
            loss_multi_instruct = loss_multi_instruct.view(-1)
            if args.use_reweight:
                # focal term
                p = torch.nn.functional.softmax(shift_logits, dim=-1)
                all_rows = torch.arange(len(shift_logits))
                pt = p[all_rows, labels]
                focal_term = (1-pt)**gamma
                # print(loss_multi_instruct.shape, focal_term.shape, reweight.shape)
                loss_multi_instruct = loss_multi_instruct*focal_term
            loss_multi_instruct = torch.sum(loss_multi_instruct)/torch.sum(labels!=-100)
            
            accelerator.backward(loss_multi_instruct)
            
            cast_dtype = get_cast_dtype(args.precision)

            #### MASK GRADIENTS FOR EMBEDDINGS ####
            # Note (anas): Do not apply weight decay to embeddings as it will break this function.
            def mask_embedding(m):
                if m.weight.requires_grad:
                    zero_mask = torch.zeros_like(m.weight.grad)
                    zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
                    m.weight.grad = m.weight.grad * zero_mask

            if args.mask_lm_head:
                model.module.lang_encoder.model.embed_tokens.apply(mask_embedding)
                model.module.lang_encoder.lm_head.apply(mask_embedding)
            # def mask_embedding(m):
            #     if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
            #         zero_mask = torch.zeros_like(m.weight)
            #         zero_mask[media_token_id] = torch.ones_like(
            #             zero_mask[media_token_id]
            #         )
            #         zero_mask[endofchunk_token_id] = torch.ones_like(
            #             zero_mask[endofchunk_token_id]
            #         )
            #         zero_mask[answer_token_id] = torch.ones_like(
            #             zero_mask[answer_token_id]
            #         )
            #         m.weight.grad = m.weight.grad * zero_mask

            # model.apply(mask_embedding)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # step optimizer and log
            # if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            #     num_steps == num_batches_per_epoch - 1
            # ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                multi_instruct_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                multi_instruct_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps * args.batch_size / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "multi_instruct_samples_per_second": multi_instruct_samples_per_second,
                        "multi_instruct_samples_per_second_per_gpu": multi_instruct_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_multi_instruct": loss_multi_instruct_b.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss Multi-Instruct: {loss_multi_instruct_b.item():.3f}"
            )


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="mm_3b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default="4b-instruct"
    )
    parser.add_argument(
        "--load_from_original_checkpoint",
        type=str,
        help="path to openflamingo provided checkpoint, in .pt format",
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--mmrec_path",
        type=str,
        help="path to mmrec dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="rec",
    )
    parser.add_argument(
        "--use_semantic",
         default=False, 
         action="store_true"
    )
    parser.add_argument("--use_reweight", default=False, action="store_true")
    parser.add_argument(
        "--subset",
        type=str,
        help="subset of Amazon.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_multi_instruct", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--mask_lm_head", action="store_true")
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--single_task",
        default=False, 
        action="store_true"
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="multi_task", 
        help="multi_task | continue"
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    parser = add_data_args(parser)
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.world_size > 1:
        device_id = init_distributed_device(args)
    else:
        device_id = 0

    random_seed(args.seed)
    if args.pretrained_model_name_or_path=="3b":
        model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path=="3b-instruct":
        model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path=="4b":
        model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        cross_attn_every_n_layers=2
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path=="4b-instruct":
        model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        cross_attn_every_n_layers=2
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path=="9b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    # 测试一下哪个比较难，只输入text，用semantic id和普通id；输入both text+id，id是atomic和semantic
    # 还有只输入id
    # 1 
    # anas-awadalla/mpt-1b-redpajama-200b	
    # openflamingo/OpenFlamingo-3B-vitl-mpt1b
    # 1
    # anas-awadalla/mpt-1b-redpajama-200b-dolly
    # openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct
    # 2
    # togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    # openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct     

    # add <answer> token to tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<answer>"]}
    )
    
    tokenizer.add_tokens(
        ["rate_1", "rate_2", "rate_3", "rate_4", "rate_5"]
    )
    
    tokenizer.add_tokens(
        ["s_0", "s_1", "s_2", "s_3", "s_4"]
    )
    
    # item_tokens = [f"item_{i}" for i in range(12094)]
    if not args.use_semantic:
        if args.subset=="all":
            item_tokens = [f"item_{i}" for i in range(22738)]
        elif args.subset=="beauty":
            item_tokens = [f"item_{i}" for i in range(4167)]
        elif args.subset=="netflix":
            item_tokens = [f"item_{i}" for i in range(1870)]
        elif args.subset=="hm":
            item_tokens = [f"item_{i}" for i in range(14901)]
        tokenizer.add_tokens(
            item_tokens
        )
    else:
        item_tokens = [f"item_{i}" for i in range(512)]
        tokenizer.add_tokens(
            item_tokens
        )
        item_tokens = [f"item_last_{i}" for i in range(32)]
        tokenizer.add_tokens(
            item_tokens
        )
    
    # item_tokens = [f"item_last_{i}" for i in range(16)]
    # tokenizer.add_tokens(
    #     item_tokens
    # )
    # 789 591 977
    img_tokens = [f"img_{i}," for i in range(1024)]
    tokenizer.add_tokens(
        img_tokens
    )
    
    # item_tokens = [f"item_{j}_{i}" for i in range(40) for j in range(3)]
    # tokenizer.add_tokens(
    #     item_tokens
    # )
    
    # item_tokens = [f"item_{j}_{i}" for i in range(256) for j in range(4)]
    # tokenizer.add_tokens(
    #     item_tokens
    # )
    
    

    model.lang_encoder.resize_token_embeddings(len(tokenizer))
    
    args.tokenizer = tokenizer
    

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    device_id = args.rank % torch.cuda.device_count()

    multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=args.task)
    multi_instruct_eval_loader = get_data_rec(args, tokenizer, "mmrec", split="eval", task=args.task)
    multi_instruct_test_loader = get_data_rec(args, tokenizer, "mmrec", split="test", task=args.task)
    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    args.train_num_samples = (
        multi_instruct_loader.num_samples
        if args.train_num_samples is None
        else args.train_num_samples
    )

    total_training_steps = len(multi_instruct_loader) * args.num_epochs

    resume_from_epoch = 0
    # check if a checkpoint exists for this run
    args.external_save_dir = (
        os.path.join(args.external_save_dir, args.run_name)
        if args.external_save_dir
        else args.run_name
    )
    if (
        os.path.exists(f"{args.external_save_dir}")
        and args.resume_from_checkpoint is True
    ):
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}."
            )

        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = (
        total_training_steps * args.warmup_steps_ratio
        if args.warmup_steps_ratio is not None
        else args.warmup_steps
    )

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps//args.gradient_accumulation_steps,
            num_training_steps=total_training_steps//args.gradient_accumulation_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    if args.single_task:
        model, optimizer, lr_scheduler, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader   = accelerator.prepare(
            model, optimizer, lr_scheduler, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader
        )
    else:
        multi_instruct_test_loader_img_sel = get_data_rec(args, tokenizer, "mmrec", split="test", task="img_sel")
        multi_instruct_test_loader_search = get_data_rec(args, tokenizer, "mmrec", split="test", task="search")
        multi_instruct_test_loader_exp = get_data_rec(args, tokenizer, "mmrec", split="test", task="exp")
        multi_instruct_test_loader_rec = get_data_rec(args, tokenizer, "mmrec", split="test", task="rec")
        model, optimizer, lr_scheduler, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader_rec, multi_instruct_test_loader_exp, multi_instruct_test_loader_img_sel, multi_instruct_test_loader_search   = accelerator.prepare(
            model, optimizer, lr_scheduler, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader_rec, multi_instruct_test_loader_exp,
            multi_instruct_test_loader_img_sel, multi_instruct_test_loader_search
        )
    # multi_instruct_test_loader_img_gen
    model.train()
    device_id = accelerator.device
    if args.single_task:
        multi_instruct_test_loader = multi_instruct_test_loader
        # eval_func = eval(f"eval_model_{args.task}")
        if args.task=="rec":
            eval_func = eval_model_rec
        elif args.task=="exp":
            eval_func = eval_model_exp
        elif args.task=="img_sel":
            eval_func = eval_model_img_sel
        elif args.task=="search":
            eval_func = eval_model_search
        elif args.task=="img_gen":
            eval_func = eval_model_img_gen
        else:
            raise KeyError("Not Supported Task Type")
    interval = 1 if args.task!="img_gen" else 1
    for epoch in range(resume_from_epoch, args.num_epochs):
        # multi_instruct_dataset.set_epoch(epoch)
        if args.train_method=="continue":
            if epoch<=args.num_epochs//4:
                multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=["rec"])
                multi_instruct_loader  = accelerator.prepare(multi_instruct_loader)
            elif args.num_epochs//4<epoch<=args.num_epochs//2:
                multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=["rec", "search"])
                multi_instruct_loader  = accelerator.prepare(multi_instruct_loader)
            elif args.num_epochs//2<epoch<=args.num_epochs//4*3:
                multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=["rec", "search", "img_sel"])
                multi_instruct_loader  = accelerator.prepare(multi_instruct_loader)
            else:
                multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=["rec", "search", "img_sel", "exp"])
                multi_instruct_loader  = accelerator.prepare(multi_instruct_loader)
        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            multi_instruct_loader=multi_instruct_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

        accelerator.wait_for_everyone()
        
        if (epoch+1)%interval==0:
            if args.do_eval:
                eval_func(
                    args=args,
                    model=model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    multi_instruct_loader=multi_instruct_eval_loader,
                    accelerator=accelerator,
                    device_id=device_id,
                    wandb=wandb,
                )
                accelerator.wait_for_everyone()
            if args.do_test:
                if args.single_task:
                    # if epoch>5:
                    eval_func(
                        args=args,
                        model=model,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        multi_instruct_loader=multi_instruct_test_loader,
                        accelerator=accelerator,
                        device_id=device_id,
                        wandb=wandb,
                    )
                else:
                    # if args.train_method=="continue" and epoch<=args.num_epochs//2:
                    #     continue
                    eval_model_rec(
                        args=args,
                        model=model,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        multi_instruct_loader=multi_instruct_test_loader_rec,
                        accelerator=accelerator,
                        device_id=device_id,
                        wandb=wandb,
                    )
                    
                    # eval_model_search(
                    #     args=args,
                    #     model=model,
                    #     epoch=epoch,
                    #     tokenizer=tokenizer,
                    #     optimizer=optimizer,
                    #     lr_scheduler=lr_scheduler,
                    #     multi_instruct_loader=multi_instruct_test_loader_search,
                    #     accelerator=accelerator,
                    #     device_id=device_id,
                    #     wandb=wandb,
                    # )
                    
                    eval_model_exp(
                        args=args,
                        model=model,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        multi_instruct_loader=multi_instruct_test_loader_exp,
                        accelerator=accelerator,
                        device_id=device_id,
                        wandb=wandb,
                    )
                    
                    # eval_model_img_sel(
                    #     args=args,
                    #     model=model,
                    #     epoch=epoch,
                    #     tokenizer=tokenizer,
                    #     optimizer=optimizer,
                    #     lr_scheduler=lr_scheduler,
                    #     multi_instruct_loader=multi_instruct_test_loader_img_sel,
                    #     accelerator=accelerator,
                    #     device_id=device_id,
                    #     wandb=wandb,
                    # )
                    
                    # eval_model_img_gen(
                    #     args=args,
                    #     model=model,
                    #     epoch=epoch,
                    #     tokenizer=tokenizer,
                    #     optimizer=optimizer,
                    #     lr_scheduler=lr_scheduler,
                    #     multi_instruct_loader=multi_instruct_test_loader_img_sel,
                    #     accelerator=accelerator,
                    #     device_id=device_id,
                    #     wandb=wandb,
                    # )
            
                accelerator.wait_for_everyone()
            if args.rank == 0:
                if not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir)

                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    get_checkpoint(model=unwrapped_model),
                    f"{args.external_save_dir}/weights_epoch_{epoch}.pt",
                )

    accelerator.wait_for_everyone()
    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            get_checkpoint(model=unwrapped_model),
            f"{args.external_save_dir}/final_weights.pt",
        )
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.external_save_dir}/final_weights.pt")


if __name__ == "__main__":
    main()
