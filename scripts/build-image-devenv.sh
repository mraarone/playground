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
    -t "$GITHUB_REPOSITORY-devenv:latest" \
    $( (( $WILL_PUSH == 1 )) && printf %s '--push' ) \
    -f dockerfiles/Dockerfile.devenv \
    .
