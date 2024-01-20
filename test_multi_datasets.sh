#!/bin/bash
sed -i "s/classify_DB_csv_./classify_DB_csv_1/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml >log_TEST_csv_1.txt 2>err_csv_1.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_1"

sed -i "s/classify_DB_csv_./classify_DB_csv_2/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml >log_TEST_csv_2.txt 2>err_csv_2.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_2"

sed -i "s/classify_DB_csv_./classify_DB_csv_3/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml >log_TEST_csv_3.txt 2>err_csv_3.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_3"

sed -i "s/classify_DB_csv_./classify_DB_csv_4/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml >log_TEST_csv_4.txt 2>err_csv_4.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_4"

sed -i "s/classify_DB_csv_./classify_DB_csv_5/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml >log_TEST_csv_5.txt 2>err_csv_5.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_5"

