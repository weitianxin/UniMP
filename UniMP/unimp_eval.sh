
lr=$1
bsz=$2
modelname=$3
subset=$4
task=$5
epoch=$6
gamma=$7
nsteps=$8
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 60001 mmrec_eval.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path=../data/processed_filter_8_all/ \
--subset=${subset} \
--batch_size=${bsz} \
--num_epochs=50 \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--load_weights_name="weights_epoch_${epoch}.pt" \
--run_name="mmrec-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-cosine-${modelname}-b${bsz}-filter8-${subset}-epoch10" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--task=${task} \
--do_test

# --report_to_wandb \
# mmrec-personalize-${task}-${lr}-cosine-${modelname}-b${bsz}-filter 
# gen_processed_filter_5_5_cloth


