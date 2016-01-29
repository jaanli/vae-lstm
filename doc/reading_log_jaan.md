
# 2016-01-29 bengio et al 2003

i enjoyed this overview of probabilistic word vector models.

discussion points:

* word embeddings are typically used as input to e.g. a softmax classifier. this is problematic when we have a large vocabulary (e.g. 1M articles on the arxiv.)
* to solve the problem of needing a hierarchical softmax, why not try to predict the N-dimensional real-valued embeddings directly?
* this mixes model and prediction. how can we justify this probabilistically?