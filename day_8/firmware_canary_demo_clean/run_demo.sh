#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main_v4_full.py
