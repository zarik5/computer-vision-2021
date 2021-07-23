#!/bin/bash

cd build
make
cd ..

for file in data/Kaggle_ships/*
do
    command="build/final_project sea_segment out/model.pb $file out data/Kaggle_ships_mask/`basename $file`"
    echo $command
    $command
done

for file in data/venice_dataset/*
do
    command="build/final_project sea_segment out/model.pb $file out data/venice_dataset_mask/`basename $file`"
    echo $command
    $command
done