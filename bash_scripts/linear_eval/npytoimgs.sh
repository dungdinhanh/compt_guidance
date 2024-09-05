#!/bin/bash

classes=("1" "218" "263" "264" "277" "278" "279" "281" "282" "353")

scales=("1.5" "2.0" "4.0" "7.0")

skips=("1")


for scale in "${scales[@]}"
do
  for class in "${classes[@]}"
  do
    for skip in "${skips[@]}"
    do
      cmd="python scripts_gdiff/npytoimgs.py  --numpy runs/analayse/revise_images/DiT/normal/IMN256/class${class}/scale${scale}_sk${skip}/reference/samples_80x256x256x3.npz --rgb"
      echo ${cmd}
      eval ${cmd}
      done
      done
      done

#scales=("1.5" "2.0" "4.0" "7.0")

scales=("6.0" "8.0" "10.0" "12.0")
skips=("11")

for scale in "${scales[@]}"
do
  for class in "${classes[@]}"
  do
    for skip in "${skips[@]}"
    do
      cmd="python scripts_gdiff/npytoimgs.py  --numpy runs/analayse/revise_images/DiT/compact/IMN256/class${class}/scale${scale}_sk${skip}/reference/samples_80x256x256x3.npz --rgb"
      echo ${cmd}
      eval ${cmd}
      done
      done
      done
