#!/bin/bash

set -e
source /opt/venv/bin/activate
echo "$@"
# split args
exec "$@"

