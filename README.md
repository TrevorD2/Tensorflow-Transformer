# Tensorflow-Transformer
Inspired by the 2017 paper "Attention Is All You Need" this repository contains a simple Transformer architecture and examples for sequence-to-label and sequence-to-sequence tasks.

# Architecture
The entire Transformer architecture contains an encoder and decoder module, each containing attention mechanisms for sequence processing. The encoder module takes the context tokens as input and outputs a set of learned vector representations. The decoder module takes both the vector representations produced by the encoder and the previously generated tokens by the decoder in order to output a discrete probability distribution representing the probability of a given output token. In this implementation, the output of the decoder is the logits, instead of the probability distribution. Note that for both of the following examples, hyperparamers should be tuned more finely to better fit particular performance needs.

# Transformer Architecture for Sentiment Analysis
The Transformer architecture can be used for both sequence-to-label and sequence-to-sequence tasks. In the case of Sentiment Analysis (a sequence-to-label task), the encoder module is used instead of the entire architecture, along with a global pooling layer and added dense layers. This implementation is performed on the IMDB movie reviews dataset. An implementation using LSTMs instead of a Transformer can be seen in this repository: https://github.com/TrevorD2/Sentiment-Analysis-RNN. 

# Transformer Architecture for Machine Translation
Both the encoder and decoder modules of the Transformer architecture should be used for sequence-to-sequence tasks. In this repository, the model is trained on an English-to-French dataset for machine translation. 
