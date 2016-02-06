#!/bin/bash

DATE=`date +%Y-%m-%d`

if [[ "$USER" == "jaan" ]];
then
  OUT_DIR="/home/jaan/projects/vae-lstm/fit/$DATE/"
  TENSORFLOW_BASE=/home/jaan/installations/tensorflow
else
  OUT_DIR=/Users/jaanaltosaar/projects/vae-lstm/fit/$DATE/
  TENSORFLOW_BASE=/Users/jaanaltosaar/projects/installations/tensorflow
fi

mkdir $OUT_DIR

$TENSORFLOW_BASE/bazel-bin/tensorflow/models/embedding/word2vec_optimized_save_hdf5 \
  --train_data=/home/dliang/data/arxiv/dataset_2012_time_exchangeable/train.txt \
  --eval_data=/home/jaan/projects/vae-lstm/dat/questions-words.txt \
  --save_path=$OUT_DIR \
  --interactive=True
