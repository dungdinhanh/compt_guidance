#!/bin/bash

#PBS -q gpuvolta
#PBS -P zg12
#PBS -l walltime=48:00:00
#PBS -l mem=128GB
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=128GB
#PBS -l wd
#PBS -l storage=scratch/zg12
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_nci2/compt_128cond_scale2p0_log4.txt
#PBS -e output_nci2/compt_128cond_scale2p0_error4.txt


module load use.own
module load python3/3.9.2
module load gdiff
#module load ASDiffusion

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 \
--learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True\
 --use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 80 --num_samples 50000 --timestep_respacing 250 "

base_folder="/scratch/zg12/dd9648"
#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "2.0" "4.0" "6.0"  )
##scales=( "10.0"  )
#scales=( "1.0"  )
scales=("2.0")
skips=("6")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29535 MARSV2_WHOLE_LIFE_STATE=0 python3 scripts_gdiff/compt_guidance/classifier_compt_sample.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS --classifier_path models/128x128_classifier.pt \
 --logdir runs/sampling_compt/IMN128/conditional/scale${scale}_skip${skip}/ \
  --save_imgs_for_visualization True --base_folder ${base_folder} --skip ${skip}"
echo ${cmd}
eval ${cmd}
done
done

for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python3 evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_imagenet128_labeled.npz \
 ${base_folder}/runs/sampling_compt/IMN128/conditional/scale${scale}_skip${skip}/reference/samples_50000x128x128x3.npz"
echo ${cmd}
eval ${cmd}
done
done



#cmd="python scripts_hfai_gdiff/classifier_compt_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}