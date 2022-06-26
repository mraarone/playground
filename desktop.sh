#!/bin/env bash

#	-v $(pwd):/app \

docker run -it --rm --gpus all \
	-u $(id -u):$(id -g) \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/sudoers:/etc/sudoers \
	tensorflow/tensorflow
