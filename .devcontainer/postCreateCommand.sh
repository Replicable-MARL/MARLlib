#!/usr/bin/env bash

# Install MARLlib as editable python package
cd /workspaces/MARLlib && pip install -e .
pip install protobuf==3.20
pip install pytest pytest-cov