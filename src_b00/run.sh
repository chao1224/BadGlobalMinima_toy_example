#!/usr/bin/env bash

mkdir -p ../output/b00_pre_train
mkdir -p ../plotting/b00_pre_train
rm ../plotting/b00_pre_train/*
python pre_train.py > ../output/b00_pre_train/main.out
cp pre_train_model.pt ../output/b00_pre_train

mkdir -p ../output/b00_fine_tuning
mkdir -p ../plotting/b00_fine_tuning
rm ../plotting/b00_fine_tuning/*
python main.py --epoch=1000 > ../output/b00_fine_tuning/main.out
