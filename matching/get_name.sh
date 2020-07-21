#!/usr/bin/env bash
# for matchingSAD.py
# extract image's name from its path
echo "$*" | awk -F [./] '{print $(NF-1)}'