2016-02-19 arora et al., levy and goldberg, le and mikolov.

i liked levy and goldberg's paper for their clear exposition of the original word2vec model as a simple SVD matrix factorization of the shifted pointwise mutual info matrix (shifted due to sparsity, SVD for simplicity). arora et al was an interesting idea for seeing how word meaning can be multimodal through their topic-model-like notion of 'meaning atoms'. the paragraph vector idea is simple: include a paragraph/document vector in predicting words given their context window. however, it was unsatisfying that they simply concatenated the two learned paragraph vectors in their experiments.

discussion questions:
* how to discover things other than analogies. this seems combinatorially hard
* how to generate tests like analogies automatically?
* does the entropy/other metric of a word vector reveal anything about the method (svd, cbow, skip-gram approaches) in use?
* how can we learn to include higher-level sequential information through some mixture model?

2016-02-04 mikolov et al, pennington et al, arora et al.

i don't really like mikolov's paper. it obfuscates many insights (like the word context matrix) from the reader and makes it harder to understand word embeddings. however, it is important work. the glove paper is a nice contrast and much easier to read. it clearly elucidates the log-bilinear form of the model and makes clear the probabilistic reasoning behind it. the glove model is very intuitive, but i wish it had more theory - it only gives heuristics for why we should believe the log-bilinear form. the final paper (arora) is a satisfying conclusion to word vectors: they extend mnih & hinton's 2007 log-bilinear model into time through a random walk on a 'discourse space'. it is satisfying because it provides a generative process and derives the continuous bag of words and glove models as simplifications of it.

discussion points:
* what are the downsides of continuous spaces? that they are good for analogies means that they must be bad for... antonyms? searching for words *farthest* in meaning from each other? 

2016-01-29 bengio et al 2003

i enjoyed this overview of probabilistic word vector models.

discussion points:

* word embeddings are typically used as input to e.g. a softmax classifier. this is problematic when we have a large vocabulary (e.g. 1M articles on the arxiv.)
* to solve the problem of needing a hierarchical softmax, why not try to predict the N-dimensional real-valued embeddings directly?
* this mixes model and prediction. how can we justify this probabilistically?