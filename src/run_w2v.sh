#!/bin/bash

DATE=`date +%Y-%m-%d`
TENSORFLOW_BASE=/Users/jaanaltosaar/projects/installations/tensorflow
OUT_DIR=/Users/jaanaltosaar/projects/math2vec/fit/$DATE/

mkdir $OUT_DIR

# $TENSORFLOW_BASE/bazel-bin/tensorflow/models/embedding/word2vec_optimized \
#   --train_data=/Users/jaanaltosaar/projects/arxiv/dat/arxmliv/corpus_for_word2vec/arxmliv_no_inline_math \
#   --eval_data=/Users/jaanaltosaar/projects/installations/tensorflow/questions-words.txt \
#   --save_path=$OUT_DIR \
#   --interactive=True

$TENSORFLOW_BASE/word2vec_optimized \
  --train_data=/Users/jaanaltosaar/projects/math2vec/dat/modern_small_94_thru_95_preprocessed/arxmliv_no_inline_math \
  --eval_data=/Users/jaanaltosaar/projects/installations/tensorflow/questions-words.txt \
  --save_path=$OUT_DIR \
  --interactive=True