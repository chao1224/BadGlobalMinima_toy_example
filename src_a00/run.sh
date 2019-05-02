#!/usr/bin/env bash

mkdir -p ../output/a00
mkdir -p ../plotting/a00
rm ../plotting/a00/*

python main.py --epoch=1000 > ../output/a00/main.out