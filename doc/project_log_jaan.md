2016-02-04

i helped dawen get set up with tensorflow and run a vanilla lstm model. i finagled bazel (google's build system to let us compile fast word2vec c++ models) onto the GPU machine. i cleaned the dataset further and compiled it into a format accessible by the word2vec model i am using.

2016-01-29

dawen and i ran the variational LSTM specified in the /doc folder on the arxiv dataset

we visualized sample trajectories, but they didn't make a lot of sense (an arxiv doc on black holes followed by a k-means paper - huh?)

next we will run word2vec to get a baseline of what to expect of user trajectories in article space.