""" Main training script """

import argparse
import glob
import os
import random

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
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    CLIPImageProcessor,
)
import warnings
warnings.filterwarnings("ignore")

from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)

from flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from flamingo.configuration_flamingo import FlamingoConfig
from otter.modeling_otter import OtterForConditionalGeneration
from otter.configuration_otter import OtterConfig
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
        default="otter_9b",
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
        default=None,
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
        "--subset",
        type=str,
        help="subset of Amazon.",
    )
    parser.add_argument(
        "--load_weights_name",
        type=str,
        default="weights_epoch_4.pt",
        help="weights name to load",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="rec")
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
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mm_model", type=str, default="flamingo")
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
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
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

    if args.pretrained_model_name_or_path is not None:
        # config = FlamingoConfig.from_json_file("./flamingo/config.json")
        # model = FlamingoForConditionalGeneration(config)
        if args.mm_model=="flamingo":
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                # device_map="balanced"        
                )
        elif args.mm_model=="otter":
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                # device_map="balanced"        
                )
        else:
            raise KeyError("Not Supported Model Type")

    else:
        config = FlamingoConfig.from_json_file("./flamingo/config.json")
        model = FlamingoForConditionalGeneration(config=config)

        """
        TODO: deprecate this option since the original checkpoints are not supported in future versions
        TODO: all future checkpoints (even released from openflamingo), we will convert them and save to huggingface format.
        TODO: supposedly using "args.pretrained_model_name_or_path" should be the best way to load the model.
        """
        if args.load_from_original_checkpoint is not None:
            print(f"Loading checkpoint from {args.load_from_original_checkpoint}")
            model.load_state_dict(
                torch.load(args.load_from_original_checkpoint, map_location="cpu"),
                False,
            )

    tokenizer = model.text_tokenizer

    # add <answer> token to tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
    )
    tokenizer.add_tokens(
        ["rate_1", "rate_2", "rate_3", "rate_4", "rate_5"]
    )
    
    tokenizer.add_tokens(
        ["s_0", "s_1", "s_2", "s_3", "s_4"]
    )
    
    item_tokens = [f"item_{i}" for i in range(12094)]
    tokenizer.add_tokens(
        item_tokens
    )
    
    model.lang_encoder.resize_token_embeddings(len(tokenizer))

    args.tokenizer = tokenizer

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    device_id = args.rank % torch.cuda.device_count()

    multi_instruct_loader = get_data_rec(args, tokenizer, "mmrec", split="train", task=args.task)
    multi_instruct_test_loader = get_data_rec(args, tokenizer, "mmrec", split="test", task=args.task)
    multi_instruct_eval_loader = get_data_rec(args, tokenizer, "mmrec", split="eval", task=args.task)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )


    args.external_save_dir = (
        os.path.join(args.external_save_dir, args.run_name)
        if args.external_save_dir
        else args.run_name
    )
    # check if a checkpoint exists for this run
    if (
        os.path.exists(f"{args.external_save_dir}")
    ):
        resume_from_checkpoint_path = os.path.join(args.external_save_dir, args.load_weights_name)
        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, False)
    else:
        raise KeyError(f"No such folder {args.external_save_dir} of checkpoint")


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
    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    model, optimizer, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader = accelerator.prepare(
        model, optimizer, multi_instruct_loader, multi_instruct_eval_loader, multi_instruct_test_loader
    )

    
    model.eval()
    device_id = accelerator.device
    if args.task=="rec":
        # eval_model_rec(
        #     args=args,
        #     model=model,
        #     epoch=0,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=None,
        #     multi_instruct_loader=multi_instruct_eval_loader,
        #     accelerator=accelerator,
        #     device_id=device_id,
        #     wandb=wandb,
        # )
        eval_model_rec(
            args=args,
            model=model,
            epoch=0,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=None,
            multi_instruct_loader=multi_instruct_test_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
    elif args.task=="exp":
        # eval_model_exp(
        #     args=args,
        #     model=model,
        #     epoch=0,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=None,
        #     multi_instruct_loader=multi_instruct_eval_loader,
        #     accelerator=accelerator,
        #     device_id=device_id,
        #     wandb=wandb,
        #     eval_embed=True
        # )
        eval_model_exp(
            args=args,
            model=model,
            epoch=0,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=None,
            multi_instruct_loader=multi_instruct_test_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
            eval_embed=True
        )
    elif args.task=="img_sel":
        # eval_model_img_sel(
        #     args=args,
        #     model=model,
        #     epoch=0,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=None,
        #     multi_instruct_loader=multi_instruct_eval_loader,
        #     accelerator=accelerator,
        #     device_id=device_id,
        #     wandb=wandb,
        # )
        eval_model_img_sel(
            args=args,
            model=model,
            epoch=0,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=None,
            multi_instruct_loader=multi_instruct_test_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
    elif args.task=="search":
        # eval_model_img_sel(
        #     args=args,
        #     model=model,
        #     epoch=0,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=None,
        #     multi_instruct_loader=multi_instruct_eval_loader,
        #     accelerator=accelerator,
        #     device_id=device_id,
        #     wandb=wandb,
        # )
        eval_model_search(
            args=args,
            model=model,
            epoch=0,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=None,
            multi_instruct_loader=multi_instruct_test_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
    accelerator.wait_for_everyone()



if __name__ == "__main__":
    main()
