#!/bin/bash

echo "Building docker image"
docker build --platform linux/amd64 --squash -t 339713152729.dkr.ecr.us-east-2.amazonaws.com/rockdetection -f docker/Dockerfile .
