#!/usr/bin/env bash

mkdir -p ../output/a00
mkdir -p ../plotting/a00
rm ../plotting/a00/*

python main.py --lr=0.01 --epoch=50000 > ../output/a00/main.out
