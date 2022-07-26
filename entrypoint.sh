#!/bin/bash

pip install -r requirements.txt

. .venv/bin/activate

exec "$@"

