
lr=$1
bsz=$2
modelname=$3
subset=$4
task=$5
gamma=$6
nsteps=$7
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 59997 mmrec_prefix.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path="../data/processed_filter_5_${subset}/" \
--subset=${subset} \
--batch_size=${bsz} \
--task=${task} \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--num_epochs=40 \
--use_reweight \
--lr_scheduler=constant \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--load_run_name="mmrec-reweight-2-2-answer-nonsemantic-2e-4-cosine-4b-instruct-b3-filter8-all-epoch10" \
--load_weights_name="weights_epoch_7.pt" \
--run_name="mmrec-domain-${task}-all-prefix-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-constant-${modelname}-b${bsz}-filter8-${subset}" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--do_test \
--single_task \
--report_to_wandb \