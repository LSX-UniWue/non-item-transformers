#!/bin/bash

echo "copying data to cache"
cp -r /scratch/jane-doe-framework/datasets/ml-1m/ml-1m_5_0_5/ /ssd/ml-1m

echo "done copying data to cache"
ls -la /ssd
ls -la /ssd/ml-1m