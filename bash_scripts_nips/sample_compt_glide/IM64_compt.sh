#!/bin/bash



MODEL_FLAGS=""

SAMPLE_FLAGS="--batch_size 100 --num_samples 30000 --timestep_respacing 250"
SAMPLE_FLAGS="--batch_size 2 --num_samples 6 --timestep_respacing 250"


cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.5" "1.0" "2.0" )
scales=( "0.1" "0.5" "0.7" )
scales=( "0.1"  )



for scale in "${scales[@]}"
do
cmd="python scripts_glide/compt/glide_sample.py $MODEL_FLAGS --guidance_scale ${scale}  $SAMPLE_FLAGS \
 --logdir runs/sampling_glide_compt/IMN64/scale${scale}/  --skip_type linear --skip 5"
echo ${cmd}
eval ${cmd}
done


for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_MSCOCO_val_64x64_squ.npz \
 runs/sampling_glide_compt/IMN64/scale${scale}/reference/samples_30000x64x64x3.npz"
#echo ${cmd}
#eval ${cmd}
done





#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}