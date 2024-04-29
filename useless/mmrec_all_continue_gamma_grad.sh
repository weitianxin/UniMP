
# 测试一下哪个比较难，只输入text，用semantic id和普通id；输入both text+id，id是atomic和semantic
# 还有只输入id
# 测试semantic id和image，不同task
# --patch-image-size 56 \
# --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 60000
# --warmup_steps_ratio=0.0 \
# --use_semantic \
# --use_reweight \
# --gradient_accumulation_steps 2\
# --mmrec_path=../data/new_processed_filter_5_cloth/ \ cloth and shoes for img generation
lr=$1
bsz=$2
modelname=$3
subset=$4
gamma=$5
nsteps=$6
accelerate launch --config_file accelerate_configs/accelerate_config_zero2.yaml --main_process_port 59999 mmrec.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path="../data/processed_filter_8_${subset}/" \
--subset=${subset} \
--batch_size=${bsz} \
--task=rec \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--num_epochs=40 \
--use_reweight \
--lr_scheduler=constant \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--run_name="mmrec-continue-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-constant-${modelname}-b${bsz}-filter8-${subset}" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--do_test \
--train_method=continue \
--report_to_wandb \
