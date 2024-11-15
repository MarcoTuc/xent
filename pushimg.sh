
#!/bin/bash

# get the name to give the new image
while getopts "t:" arg; do
  case $arg in
    t) name=$OPTARG;;
  esac
done

docker login registry.rcp.epfl.ch
docker tag $name registry.rcp.epfl.ch/tuccio/$name
docker push registry.rcp.epfl.ch/tuccio/$name



