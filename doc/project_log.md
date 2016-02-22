2016-02-19

hashed out the hierarchical timeseries model idea in /doc, related work is included in the original project plan. ran more fits of lstms. after chat with dave, decided to stop working on variational autoencoder based models for now. we will switch to a simple linear regression of word2vec features for items and see how well this does. laurent, dawen, and jaan met to discuss this new direction and plan for recsys.

2016-02-12

explored word2vec fits in t-SNE. determined that lstm model embeddings aren't good, but word2vec fits make sense: users stay around their interests.

2016-02-04

ran a vanilla lstm model, made sure we can overfit.
ran word2vec on dataset.

2016-01-29

ran variational LSTM specified in the /doc folder on the arxiv dataset
visualized sample trajectories, but they didn't make a lot of sense (an arxiv doc on black holes followed by a k-means paper - huh?)
next we will run word2vec to get a baseline of what to expect of user trajectories in article space.