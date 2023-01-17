#!/bin/bash

[ -z $1 ] && echo "it is missing script name"

CONTAINER=mindspore

sudo docker cp "$1" ${CONTAINER}:/tmp/$1

sudo docker exec ${CONTAINER} python "/tmp/$1" 
