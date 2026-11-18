#!/bin/bash

# Run unittests
uv run python3 unittest

# Explore the cuda environment
uv run scripts/check_cuda.py
