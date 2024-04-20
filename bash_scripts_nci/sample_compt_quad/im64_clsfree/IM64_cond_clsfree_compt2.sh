#!/bin/bash

#PBS -q gpuvolta
#PBS -P zg12
#PBS -l walltime=48:00:00
#PBS -l mem=64GB
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l jobfs=64GB
#PBS -l wd
#PBS -l storage=scratch/zg12
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_nci/compt64_clsfree_log2.txt
#PBS -e output_nci/compt64_clsfree_error2.txt


module load use.own
module load python3/3.9.2
module load cdiff

SAMPLE_FLAGS="--batch_size 130 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 200 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


base_folder="/scratch/zg12/dd9648"
#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "1.0"   )
#scales=( "10.0"  )
#scales=( "1.0"  )
skips=("5")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=30515 MARSV2_WHOLE_LIFE_STATE=0 python3  scripts_gdiff/compt_guidance/classifier_free_compt_sample.py $MODEL_FLAGS  \
 --model_path models/64x64_diffusion.pt --uncond_model_path models/64x64_diffusion_unc.pt \
  $SAMPLE_FLAGS  --logdir runs/sampling_compt2/IMN64_clsfree/conditional/scale${scale}_skip${skip}/ \
  --save_imgs_for_visualization True --base_folder ${base_folder} --skip ${skip}"
echo ${cmd}
eval ${cmd}
done
done

for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python3 evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_imagenet64_labeled.npz \
 ${base_folder}/runs/sampling_compt2/IMN64_clsfree/conditional/scale${scale}_skip${skip}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done
done



#cmd="python scripts_hfai_gdiff/classifier_compt_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}