#!/bin/bash

while getopts "t:" arg; do
  case $arg in
    t) name=$OPTARG;;
  esac
done

sh updatecode.sh -t $name
sh pushimg.sh -t $name
sh runcluster.sh -n $name