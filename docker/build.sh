#!/bin/bash

echo 'Building Dockerfile with image name marl:1.0'
#docker build -t marl:1.0 .
docker build --no-cache -t marl:1.0 . # build docker from nothing