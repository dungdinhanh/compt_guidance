#!/bin/bash


ARCHs=("unet64na" "resnet34" "resnet50"  "resnet152" )
ARCHs=("unet64na"  "resnet152" "unet64nas")
#ARCHs=( "unet64nas")
#ARCHs=("unet64na"  "resnet152" )
#ARCHs=("resnet152" )
#ARCHs=( "unet64nas")
#ARCHs=( "resnet34" "resnet50"  "resnet152" "densenet169" "densenet121" "densenet201" "squeezenet1_0" "squeezenet1_1" )
#SAMPLES="runs_ofs/sampling_amd_noisea_lssch_wconf/lsim_clip/IMN256/unconditional/scale10.0_eps0.94/reference/samples_50000x256x256x3.npz"
SAMPLES=$1

cmd="cd ../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

for ARCH in "${ARCHs[@]}"
do
MODEL_FLAGS="--arch ${ARCH} --batch-size 128 -m --workers 1 \
-p ${SAMPLES}  \
 -f 20 --image_size 64 "

cmd="python evaluations/imagenet_evaluator_models/main_eval_robustness_res.py ${MODEL_FLAGS}  -e "
echo ${cmd}
eval ${cmd}
done



#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}