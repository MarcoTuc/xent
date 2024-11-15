#!/bin/bash

# get the name to give the new image
while getopts "n:t:" arg; do
  case $arg in
    n) name=$OPTARG;;
    t) tag=$OPTARG;;
  esac
done

runai submit \
    -i registry.rcp.epfl.ch/tuccio/$tag \
    --node-pools v100 \
    --gpu 1 \
    --name $name \
    --existing-pvc claimname=csft-scratch,path=/home/marktas/storage \
    --interactive \
    --attach