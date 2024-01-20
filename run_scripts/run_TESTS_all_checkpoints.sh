#!/bin/bash
for chekcpoint_file in $(ls checkpoints/)
do
  sed -E -i "s/(.*CHECKPOINT_FILE_PATH:.*checkpoints\/).*/\1${chekcpoint_file}/g"\
  /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
  for NUM_SPATIAL_CROPS in 3 
  do
    sed -E -i "s/(.*NUM_SPATIAL_CROPS: ).*/\1${NUM_SPATIAL_CROPS}/g"\
    /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
    for NUM_ENSEMBLE_VIEWS in  5 
    do
      sed -E -i "s/(.*NUM_ENSEMBLE_VIEWS: ).*/\1${NUM_ENSEMBLE_VIEWS}/g"\
      /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
      python tools/run_net.py --cfg\
      /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_TEST_4gpus.yaml
      echo "Finished " $chekcpoint_file
    done
  done
done
