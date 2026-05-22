#!/bin/bash

git submodule update --init --recursive
uv venv
uv pip install -e safety-tooling
uv pip install -r safety-tooling/requirements_dev.txt

uv pip install -e safety-tooling