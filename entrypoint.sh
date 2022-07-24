#!/bin/bash

pip install -r requirements.txt

source .venv/bin/activate
exec "$@"

