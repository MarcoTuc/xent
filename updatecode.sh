#!/bin/bash

# get the name to give the new image
while getopts "t:" arg; do
  case $arg in
    t) name=$OPTARG;;
  esac
done

docker build -f docker/update_code -t $name .



