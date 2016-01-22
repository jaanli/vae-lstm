#!/bin/bash
if [[ "$USER" == "jaan" ]];
then
  OUT_DIR="/home/jaan/projects/vae-lstm/fit/2015-01-22/"
  DATA_PATH="/home/jaan/projects/vae-lstm/dat/simple-examples/data"
  PROGRESS="progress_bar=True"
else
  OUT_DIR=/Users/jaanaltosaar/projects/vae-lstm/fit/2015-01-21/
  DATA_PATH=/Users/jaanaltosaar/projects/vae-lstm/dat/dataset_2012_time_exchangeable/
  PROGRESS="progress_bar=True"
fi

python vae_lstm.py \
  --data_path=$DATA_PATH \
  --model=medium \
  --out_dir=$OUT_DIR \
  --debug=True \
  # --checkpoint_file=/Users/jaanaltosaar/projects/arxiv/fit/2015-12-13-ptb-vae-2/model.ckpt