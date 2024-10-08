#!/bin/bash


SAMPLE_FLAGS="--batch_size 80 --num_samples 80 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 200 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


base_folder="../selfsup-guidance/"
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
scales=("0.1")
#scales=("2.0")
skips=("5")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python3 scripts_gdiff/compt_guidance/classifier_compt_sample_general_autos.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS --classifier_path models/64x64_classifier.pt \
 --logdir runs/sampling_compt2t/IMN64/conditional/scale${scale}_skip${skip}/ \
  --save_imgs_for_visualization True --classifier_depth 4 --base_folder ${base_folder} --skip ${skip} --k 2"
echo ${cmd}
eval ${cmd}
done
done




#cmd="python scripts_hfai_gdiff/classifier_compt_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}