#!/bin/bash
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_5.yaml
echo "Finished 5 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_10.yaml
echo "Finished 10 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_15.yaml
echo "Finished 15 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_20.yaml
echo "Finished 20 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_25.yaml
echo "Finished 25 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_30.yaml
echo "Finished 30 EPOCSs"
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_35.yaml
echo "Finished 35 EPOCSs"
