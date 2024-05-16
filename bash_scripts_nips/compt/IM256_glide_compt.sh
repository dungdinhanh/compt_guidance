#!/bin/bash

export NCCL_P2P_DISABLE=1

MODEL_FLAGS=""

SAMPLE_FLAGS="--batch_size 25 --num_samples 25 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 22 --num_samples 44 --timestep_respacing 250"


cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


base_folder="../selfsup-guidance/"
#base_folder="./"


scales=(  "8.0")

skips=("7" )



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python scripts_glide/compt/glide_up_sample_time.py $MODEL_FLAGS --guidance_scale ${scale}  $SAMPLE_FLAGS \
 --logdir runsGLIDECOMPT/sampling_glidec2t/IMN256/scale${scale}_sk${skip}/ --skip_type linear --skip ${skip} --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done
done







#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}