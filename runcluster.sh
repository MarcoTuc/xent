# !/bin/bash

# get the name to give the new image
while getopts "n:t:" arg; do
  case $arg in
    n) name=$OPTARG;;
    t) tag=$OPTARG;;
  esac
done

if [ -z "$tag" ]; then
    tag=$name
fi


runai submit \
    --name $name \
    --image registry.rcp.epfl.ch/tuccio/$tag \
    --node-pools h100 \
    --gpu 1 \
    --environment WANDB_API_KEY=KEY \
    --run-as-uid 396376  \
    --run-as-gid 10776 \
    --existing-pvc claimname=csft-scratch,path=/home/marktas/storage \
    --command -- python experiments/pretraining/wiki_pretraining.py \
    # --interactive \
    # --attach