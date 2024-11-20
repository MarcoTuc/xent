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
    --node-pools default \
    --gpu 1 \
    --environment WANDB_API_KEY=e37d5bd5c2f36a7a6b7eb1c2501056a9884b4db2 \
    --run-as-uid 396376  \
    --run-as-gid 10776 \
    --existing-pvc claimname=csft-scratch,path=/home/marktas/storage \
    --command -- python experiments/crosstalk/ranking_base.py \
    # --interactive \
    # --attach