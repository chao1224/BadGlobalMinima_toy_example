#!/usr/bin/env bash

mkdir -p ../output/c07_fine_tuning
mkdir -p ../plotting/c07_fine_tuning
rm ../plotting/c07_fine_tuning/*
python main.py > ../output/c07_fine_tuning/main.out
