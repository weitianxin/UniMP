lr=$1
bsz=$2
modelname=$3
subset=$4
gamma=$5
nsteps=$6
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 59999 mmrec.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path="../data/${subset}/" \
--subset=${subset} \
--batch_size=${bsz} \
--task=rec \
--use_reweight \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--num_epochs=50 \
--lr_scheduler=constant \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--run_name="mmrec-hm-len820-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-constant-${modelname}-b${bsz}-${subset}" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--do_test \
--single_task \
--report_to_wandb \
