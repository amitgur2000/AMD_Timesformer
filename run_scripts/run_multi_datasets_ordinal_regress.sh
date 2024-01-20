#!/bin/bash
sed -i "s/classify_DB_csv_./classify_DB_csv_4/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml >log_csv_4_OR.txt 2>err_csv_4_OR.txt
sed -i "s/classify_DB_csv_./classify_DB_csv_4/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails_2/TEST_*
source run_TESTS_2_5-30.sh >log_csv_4_OR_TEST.txt 2>err_csv_4_OR_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_4 OR"

sed -i "s/classify_DB_csv_./classify_DB_csv_5/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml >log_csv_5_OR.txt 2>err_csv_5_OR.txt
sed -i "s/classify_DB_csv_./classify_DB_csv_5/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails_2/TEST_*
source run_TESTS_2_5-30.sh >log_csv_5_OR_TEST.txt 2>err_csv_5_OR_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_5 OR"

sed -i "s/classify_DB_csv_./classify_DB_csv_6/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml >log_csv_6_OR.txt 2>err_csv_6_OR.txt
sed -i "s/classify_DB_csv_./classify_DB_csv_6/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails_2/TEST_*
source run_TESTS_2_5-30.sh >log_csv_6_OR_TEST.txt 2>err_csv_6_OR_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/checkpoints/*
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_6 OR"

sed -i "s/classify_DB_csv_./classify_DB_csv_7/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml
python tools/run_net.py --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_ordinal_regress_4gpus.yaml >log_csv_7_OR.txt 2>err_csv_7_OR.txt
sed -i "s/classify_DB_csv_./classify_DB_csv_7/g" /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/tmp_trails_2/TEST_*
source run_TESTS_2_5-30.sh >log_csv_7_OR_TEST.txt 2>err_csv_7_OR_TEST.txt
rm /home/agur/projects/AMD/AMD_Timesformer/tmp_videos/*
rm /home/agur/projects/AMD/AMD_Timesformer/stdout.log
echo "Finished classify_DB_csv_7 OR"

