#!/bin/bash

if [[ -z "${MODEL_NAME}" ]]; then
	echo "MODEL_NAME must be set"
	exit 1
fi
python3 -u download_model.py
python3 -u handler.py