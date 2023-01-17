#!/bin/bash

[ -z $1 ] && echo "it is missing file name"

CONTAINER=mindspore

sudo docker cp "$1" ${CONTAINER}:/tmp/$1
