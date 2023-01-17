#!/bin/bash

[ -z $1 ] && echo "it is missing script name"

CONTAINER=mindspore


echo "Warning: It is considered that '$1' is already available in ${CONTAINER} /tmp"
echo "Output:"
echo ""
sudo docker exec ${CONTAINER} python "/tmp/$1"
echo "Done!"
