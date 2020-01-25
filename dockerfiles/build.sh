#!/bin/bash
USER=hungpham2511
IMAGE=toppra-dep
VERSION=0.0.3

echo "Building docker image: $USER/$IMAGE:$VERSION"
docker build -t ${IMAGE} .

docker tag ${IMAGE} ${USER}/${IMAGE}:${VERSION}
docker tag ${IMAGE} ${USER}/${IMAGE}:latest
