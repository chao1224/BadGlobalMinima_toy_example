#!/usr/bin/env bash

cd src_a00

python reload_main.py

cd ..

cd src_b00
python reload_pre_train.py
python reload_main.py
cd ..

cd src_c07
python reload_main.py
cd ..
