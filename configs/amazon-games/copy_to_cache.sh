#!/bin/bash

echo "copying data to cache"
cp -r /scratch/jane-doe-framework/datasets/amazon/games /ssd/

echo "done copying data to cache"
ls -la /ssd
ls -la /ssd/games