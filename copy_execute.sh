#!/bin/bash

[ -z $1 ] && echo "it is missing script name"

CONTAINER=mindspore

echo "Copying '$1' in ${CONTAINER} /tmp folder"

sudo docker cp "$1" ${CONTAINER}:/tmp/$1

echo "Executing '$1'..."
echo ""
sudo docker exec ${CONTAINER} python "/tmp/$1" 
echo "Done!"

