#!/usr/bin/env bash

mkdir -p ../output/c07_fine_tuning
mkdir -p ../plotting/c07_fine_tuning
rm ../plotting/c07_fine_tuning/*
python main.py --epoch=1000 --lr=0.01  --batch-size=10 > ../output/c07_fine_tuning/main.out
