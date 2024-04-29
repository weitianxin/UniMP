lr=$1
task=$2
bsz=$3
modelname=$4
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 60002 mmrec.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path=../data/gen_processed_filter_5_5_cloth/ \
--subset=cloth \
--batch_size=${bsz} \
--num_epochs=10 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--run_name="mmrec-personalize-${task}-${lr}-cosine-${modelname}-b${bsz}-filter" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--task=${task} \
--do_test \
--report_to_wandb \
--single_task \

