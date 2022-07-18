#!/bin/bash

set -e

pwd
ls

if [ $1 == "--push" ]; then
    WILL_PUSH=1
else
    WILL_PUSH=0
fi

docker buildx build \
    --platform linux/amd64,linux/arm64,linux/arm/v7,linux/arm/v6 \
    -t "$GITHUB_REPOSITORY-devenv:latest" \
    $( (( $WILL_PUSH == 1 )) && printf %s '--push' ) \
    -f dockerfiles/Dockerfile.devenv \
    .
