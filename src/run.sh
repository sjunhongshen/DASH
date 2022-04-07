#!/bin/bash

DS="CIFAR100" # SPHERICAL DARCY-FLOW-5 PSICOV COSMIC NINAPRO FSD ECG SATELLITE DEEPSEA MNIST MUSIC

for SEED in 0 1 2 ; do
    
    for i in $DS ; do
        # python3 -W ignore main.py --dataset $i --baseline 1 --seed $SEED
        python3 -W ignore main.py --dataset $i --experiment_id 0 --seed $SEED --valid_split 0
    done

done

# speed test
# python3 speed.py --experiment_id 0
# python3 speed.py --experiment_id 0 --test_input_size 1
