#!/bin/bash
source run_test.sh checkpoints/model_29.pth
echo "Finished 1"
source run_test.sh checkpoints/model_59.pth
echo "Finished 2"
source run_test.sh checkpoints/model_89.pth
echo "Finished 3"
source run_test.sh checkpoints/model_119.pth
echo "Finished 4"
source run_test.sh checkpoints/model_149.pth
echo "Finished 5"
source run_test.sh checkpoints/model_179.pth
echo "Finished 6"
source run_test.sh checkpoints/checkpoint.pth
echo "Finished 7"
