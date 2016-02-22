# 2016-02-19

hashed out the hierarchical timeseries model idea in /doc, related work is included in the original project plan. ran more fits of lstms. after chat with dave, decided to stop working on variational autoencoder based models for now. we will switch to a simple linear regression of word2vec features for items and see how well this does. laurent, dawen, and jaan met to discuss this new direction and plan for recsys.

#2016-02-11

TSNE embedding of word2vec makes perfect sense (small clusters roughly by subject) and actual user trajectories look reasonable (again makes sense because the quality of the embedding). Vanilla LSTM isn't as good, but this embedding is only the input. 

For evaluation:
- [Session-based Recommendations with Recurrent Neural Networks](http://arxiv.org/abs/1511.06939) uses entire sequence to test: Each time a new item is presented to the model and we check the rank of the next item. However, it is not clear how to do this with word2vec -- maybe we don't need to? Another problem with this evaluation is that we can no longer compare with traditional CF model. A workaround proposed in this paper is to use the average of item feature vectors of the items that had occurred in the sequence so far as the user feature vector and optimize whatever loss function (e.g. BPR) with SGD. 
- [TribeFlow: Mining & Predicting User Trajectories](http://arxiv.org/abs/1511.01032) uses the first part of the entire trajectory as training and tests on the rest, so that user representation can be learned first. In this setting, we can compare with traditional CF model more easily. 

Right now our data is split in the first way. Ultimately we would like to evaluate under both settings. 

#2016-02-04

ran a vanilla lstm model, made sure we can overfit.
ran word2vec on dataset.

#2016-01-29

ran variational LSTM specified in the /doc folder on the arxiv dataset
visualized sample trajectories, but they didn't make a lot of sense (an arxiv doc on black holes followed by a k-means paper - huh?)
next we will run word2vec to get a baseline of what to expect of user trajectories in article space.

