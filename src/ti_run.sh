module load python
module load cudatoolkit/7.0 cudann

python vae_lstm.py \
  --data_path=/home/altosaar/projects/arxiv/dat/simple-examples/data/ \
  --model=small \
  --out_dir=/home/altosaar/projects/arxiv/fit/2015-12-15-ptb-small/ \
  # --checkpoint_dir=/home/altosaar/projects/arxiv/fit/2015-12-13-ptb-vae-2/
