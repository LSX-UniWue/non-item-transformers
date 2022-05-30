#!/usr/bin/env bash
# merges three yaml files with values later listed files taking precedence. *d enables deep array merging
yq eval-all 'select(fileIndex == 0) *d select(fileIndex == 1) *d select(fileIndex == 2)' file0.yaml file1.yaml file2.yaml