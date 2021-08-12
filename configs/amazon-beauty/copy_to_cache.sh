#!/bin/bash

echo "copying data to cache"
cp -r /scratch/jane-doe-framework/datasets/amazon/beauty /ssd/

echo "done copying data to cache"
ls -la /ssd
ls -la /ssd/beauty