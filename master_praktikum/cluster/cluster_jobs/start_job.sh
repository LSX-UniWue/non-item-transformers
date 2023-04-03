#!/bin/bash

kubectl -n studbonda create -f /home/stud/bonda/master_praktikum/cluster_jobs/$1.yaml

echo "done"