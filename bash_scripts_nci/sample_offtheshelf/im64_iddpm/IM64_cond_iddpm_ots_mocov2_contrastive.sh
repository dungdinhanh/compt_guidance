#!/bin/bash

#PBS -q gpuvolta
#PBS -P zg12
#PBS -l walltime=24:00:00
#PBS -l mem=32GB
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l jobfs=50GB
#PBS -l wd
#PBS -l storage=scratch/zg12
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_nci/log_ddpm1.txt
#PBS -e output_nci/error_ddpm1.txt

module load use.own
module load python3/3.9.2
module load gdiff
#module load ASDiffusion

nvidia-smi

SAMPLE_FLAGS="--batch_size 350 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 1 --num_samples 2 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True \
--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"

#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "0.0"  "0.5" "1.0" "2.0" )
#scales=( "10.0"  )
scales=( "0.0"  )
jointtemps=( "1.0")
margintemps=( "1.0" )
storage_dir="/scratch/zg12/dd9648"

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python3 script_odiff/iddpm/iddpm_sample_ots_nci.py $MODEL_FLAGS --classifier_scale ${scale}  \
--classifier_type mocov2 --model_path ${storage_dir}/models/imagenet64_cond_270M_250K.pt $SAMPLE_FLAGS --joint_temperature ${jt} \
 --logdir ${storage_dir}/runs/sampling_ots_iddpm/IMN64/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose_sup_contrastive/ \
 --features ${storage_dir}/eval_models/imn64_mocov2/reps3.npz \
 --save_imgs_for_visualization True &>> output_nci/output_iddpm${scale}.txt"
echo ${cmd}
eval ${cmd}
done
done
done

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
cmd="python3 evaluations/evaluator_tolog.py ${storage_dir}/reference/VIRTUAL_imagenet64_labeled.npz \
 ${storage_dir}/runs/sampling_ots_iddpm/IMN64/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose_sup_contrastive/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done
done
done




#cmd="python scripts_hfai_gdiff/classifier_compt_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}