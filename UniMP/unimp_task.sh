
lr=$1
bsz=$2
modelname=$3
subset=$4
task=$5
gamma=$6
nsteps=$7
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 59999 mmrec.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path="../data/processed_filter_8_${subset}/" \
--subset=${subset} \
--batch_size=${bsz} \
--task=${task} \
--use_reweight \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--num_epochs=10 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--run_name="mmrec-single-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-cosine-${modelname}-b${bsz}-filter8-${subset}" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--do_test \
--single_task \
--report_to_wandb \
