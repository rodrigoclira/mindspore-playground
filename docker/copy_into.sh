#!/bin/bash

[ -z $1 ] && echo "it is missing file name"

CONTAINER=mindspore

echo "Copying '$1' in ${CONTAINER} /tmp folder"
sudo docker cp "$1" ${CONTAINER}:/tmp/$1
echo "Done!"
