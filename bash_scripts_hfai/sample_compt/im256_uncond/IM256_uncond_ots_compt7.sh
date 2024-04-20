#!/bin/bash



MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250 "

base_folder="./"
#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "2.0" "4.0" "6.0"  )
##scales=( "10.0"  )
#scales=( "1.0"  )
scales=("28.0" "30.0")
skips=("2" "5" "6")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python scripts_gdiff/compt_guidance/classifier_compt_sample.py \
  $MODEL_FLAGS --classifier_scale ${scale}  \
 --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --classifier_path models/256x256_classifier.pt \
 --logdir runs/sampling_compt2/IMN256/unconditional/scale${scale}_skip${skip}/ \
  --save_imgs_for_visualization True --base_folder ${base_folder} --skip ${skip}"
echo ${cmd}
eval ${cmd}
done
done

for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_imagenet256_labeled.npz \
 ${base_folder}/runs/sampling_compt2/IMN256/unconditional/scale${scale}_skip${skip}/reference/samples_50000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done
done



#cmd="python scripts_hfai_gdiff/classifier_compt_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}