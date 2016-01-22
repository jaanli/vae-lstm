OUT_DIR="/Users/jaanaltosaar/projects/arxiv/fit/2015-12-15-ptb-dbg-2/"

cp vae_lstm.py $OUT_DIR
cp run.sh $OUT_DIR

python vae_lstm.py \
  --data_path=/Users/jaanaltosaar/projects/arxiv/dat/simple-examples/data/ \
  --model=medium \
  --out_dir=$OUT_DIR \
  --debug=True \
  # --checkpoint_file=/Users/jaanaltosaar/projects/arxiv/fit/2015-12-13-ptb-vae-2/model.ckpt