#!/bin/bash

echo 'Building Dockerfile with image name marllib:1.0'
#docker build -t marllib:1.0 .
docker build --no-cache -t marllib:1.0 . # build docker from nothing