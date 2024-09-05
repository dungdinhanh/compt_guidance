#!/bin/bash

export NCCL_P2P_DISABLE=1

MODEL_FLAGS=""

SAMPLE_FLAGS="--batch_size 25 --num_samples 30000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 22 --num_samples 44 --timestep_respacing 250"


cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


base_folder="/hdd/dungda/selfsup-guidance/"
#base_folder="./"

scales=( "5.5" "6.0" "6.5" "7.0" "7.5" "8.0")
scales=(  "8.0")
scales=("9.0")
scales=(  "11.0")
scales=(  "13.0" "14.0" )
skips=("7" "8" "9" "10")
#skips=("7" "8" "9")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python scripts_glide/compt/glide_up_sample.py $MODEL_FLAGS --guidance_scale ${scale}  $SAMPLE_FLAGS \
 --logdir runsGLIDECOMPT/sampling_glide/IMN256/scale${scale}_sk${skip}/ --skip_type linear --skip ${skip} --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done
done


for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="python evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_MSCOCO_val_256x256_squ256.npz \
 ${base_folder}/runsGLIDECOMPT/sampling_glide/IMN256/scale${scale}_sk${skip}/reference/samples_30000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done
done





#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}