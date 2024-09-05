#!/bin/bash


#ARCHs=("unet256na" "resnet34" "resnet50"  "resnet152" "densenet169" "densenet121" "densenet201" "squeezenet1_0" "squeezenet1_1" )
#ARCHs=("unet256na" )
#ARCHs=("resnet18" )
SAMPLES=("runs/sampling_compt2/IMN64/conditional/scale0.1_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale0.1_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale0.3_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale0.5_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale1.0_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale2.0_skip5/reference/samples_50000x64x64x3.npz" \
"runs/sampling_compt2/IMN64/conditional/scale4.0_skip5/reference/samples_50000x64x64x3.npz" \
"/home/dzung/unisyddev/metaguidance/runs/sampling_amd/IMN64/conditional/scale0.5/reference/samples_50000x64x64x3.npz"
)

#SAMPLES=( )
#SAMPLES=("/home/dzung/unisyddev/metaguidance/runs/sampling_amd/IMN64/conditional/scale2.0/reference/samples_50000x64x64x3.npz" )


#SAMPLES=("runs/sampling_compt2/IMN64/conditional/scale0.1_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale2.2_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale2.4_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale2.6_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale2.8_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale3.4_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale3.6_skip5/reference/samples_50000x64x64x3.npz" \
#"runs/sampling_compt2/IMN64/conditional/scale3.8_skip5/reference/samples_50000x64x64x3.npz" \
#)
#
#SAMPLES=("/home/dzung/unisyddev/metaguidance/runs/sampling_amd/IMN64/conditional/scale2.0/reference/samples_50000x64x64x3.npz" )
SAMPLES=("/home/dzung/unisyddev/compt_guidance/runs/sampling_compt2/IMN64/unconditional/scale4.0/reference/samples_50000x64x64x3.npz" \
"/home/dzung/unisyddev/compt_guidance/runs/sampling_compt2/IMN64/unconditional/scale17.0_skip5/reference/samples_50000x64x64x3.npz")

SAMPLES=("/home/dzung/unisyddev/compt_guidance/runs/sampling_compt2/IMN64/unconditional/scale18.0_skip5/reference/samples_50000x64x64x3.npz" \
"/home/dzung/unisyddev/compt_guidance/runs/sampling_compt2/IMN64/unconditional/scale19.0_skip5/reference/samples_50000x64x64x3.npz")

SAMPLES=("/home/dzung/unisyddev/compt_guidance/runs/sampling_compt2_overfitting/IMN64_ES/samples_50000x64x64x3.npz" )

cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

#cmd="ls"
#echo ${cmd}
#eval ${cmd}


for sample in "${SAMPLES[@]}"
do
cmd="hfai bash bash_scripts_nips/eval_overfitting/IM64.sh ${sample} ++"
echo ${cmd}
eval ${cmd}
done
