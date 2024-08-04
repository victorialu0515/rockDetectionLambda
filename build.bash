#!/bin/bash

echo "Building docker image"
docker build --squash -t 339713152729.dkr.ecr.us-east-2.amazonaws.com/rockdetection -f docker/Dockerfile .
