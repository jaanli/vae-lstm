module load python
module load cudatoolkit/7.0 cudann

python vae_lstm.py \
  --data_path=/home/altosaar/projects/arxiv/dat/simple-examples/data/ \
  --model=medium \
  --out_dir=/home/altosaar/projects/arxiv/fit/2015-12-13-ptb-vae/ \
  # --checkpoint_dir=/home/altosaar/projects/arxiv/fit/2015-12-9-ptb-vae-lstm-small/
