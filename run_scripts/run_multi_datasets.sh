#!/bin/bash
sed -i "s/classify_DB_csv_./classify_DB_csv_4/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml >log_csv_4.txt 2>err_csv_4.txt
# sed -i "s/classify_DB_csv_./classify_DB_csv_4/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_*
# source run_TESTS_5-30.sh >log_csv_4_TEST.txt 2>err_csv_4_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_4"

sed -i "s/classify_DB_csv_./classify_DB_csv_5/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml >log_csv_5.txt 2>err_csv_5.txt
# sed -i "s/classify_DB_csv_./classify_DB_csv_5/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_*
# source run_TESTS_5-30.sh >log_csv_5_TEST.txt 2>err_csv_5_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_5"

sed -i "s/classify_DB_csv_./classify_DB_csv_6/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml >log_csv_6.txt 2>err_csv_6.txt
# sed -i "s/classify_DB_csv_./classify_DB_csv_6/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_*
# source run_TESTS_5-30.sh >log_csv_6_TEST.txt 2>err_csv_6_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_6"

sed -i "s/classify_DB_csv_./classify_DB_csv_7/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml >log_csv_7.txt 2>err_csv_7.txt
# sed -i "s/classify_DB_csv_./classify_DB_csv_7/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails/TEST_*
# source run_TESTS_5-30.sh >log_csv_7_TEST.txt 2>err_csv_7_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_7"


