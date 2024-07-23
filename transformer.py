import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout
import numpy as np

class Transformer(Model):
    def __init__(self, context_vocab_size, model_dim, max_input_size, stack_height, heads, key_dim, ff_dim, dropout, target_vocab_size=-1):
        super().__init__()
        #If the context and target vocabularies are the same, set both to context_vocab
        self.target_vocab_size = context_vocab_size if target_vocab_size==-1 else target_vocab_size 
        self.context_vocab_size = context_vocab_size
        
        #Create Transformer layers with appropriate parameters
        self.encoder = Encoder(self.context_vocab_size, model_dim, max_input_size, stack_height, heads, key_dim, ff_dim, dropout)
        self.decoder = Decoder(self.target_vocab_size, model_dim, max_input_size, stack_height, heads, key_dim, ff_dim, dropout)
        self.drop = Dropout(dropout)
        self.dense = Dense(self.target_vocab_size)

    def call(self, x):
        inputs, outputs = x

        context = self.encoder(inputs)
        context = self.drop(context)

        dense_in = self.decoder(outputs, context)
        dense_in = self.drop(dense_in)

        logits = self.dense(dense_in)

        return logits

#Encoder module of the transformer
class Encoder(Layer):
    def __init__(self, vocab_size, model_dim, max_input_size, stack_height, heads, key_dim, ff_dim, dropout):
        super().__init__()
        self.drop = Dropout(dropout)
        self.process = Preprocessing(vocab_size, model_dim, max_input_size)
        self.encoding_layers = [EncodingLayer(heads, model_dim, key_dim, ff_dim, dropout) for _ in range(stack_height)]

    #Processes input, then feeds processed input to encoder layers
    def call(self, x):
        x = self.process(x)
        x = self.drop(x)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        return x

#Decoder module of the transformer
class Decoder(Layer):
    def __init__(self, vocab_size, model_dim, max_input_size, stack_height, heads, key_dim, ff_dim, dropout):
        super().__init__()
        self.drop = Dropout(dropout)
        self.process = Preprocessing(vocab_size, model_dim, max_input_size)
        self.decoding_layers = [DecodingLayer(heads, model_dim, key_dim, ff_dim, dropout) for _ in range(stack_height)]

    #Processes input, then feeds processed input to decoder layers
    def call(self, x, context):
        x = self.process(x)
        x = self.drop(x)
        for decoding_layer in self.decoding_layers:
            x = decoding_layer(x, context)
        return x

#Single encoding layer
class EncodingLayer(Layer):
    def __init__(self, heads, model_dim, key_dim, ff_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.add_norm = AddNorm()
        self.fc1 = Dense(ff_dim)
        self.fc2 = Dense(model_dim)
    
    def call(self, input):
        global_attention_out = self.mha(
            query=input,
            value=input,
            key=input
        )
        ff_in = self.add_norm(global_attention_out, input)
        ff = self.fc1(ff_in)
        ff_out = self.fc2(ff)
        return self.add_norm(ff_out, ff_in)

#Single decoding layer
class DecodingLayer(Layer):
    def __init__(self, heads, model_dim, key_dim, ff_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.add_norm = AddNorm()
        self.fc1 = Dense(ff_dim)
        self.fc2 = Dense(model_dim)
    
    def call(self, target_input, context_input):
        masked_attention_out = self.mha(
            query=target_input,
            value=target_input,
            key=target_input,
            use_causal_mask=True
        )

        cross_attention_in = self.add_norm(masked_attention_out, target_input)

        cross_attention_out = self.mha(
            query=cross_attention_in,
            value=context_input,
            key=context_input
        )

        ff_in = self.add_norm(cross_attention_in, cross_attention_out)

        ff = self.fc1(ff_in)
        ff_out = self.fc2(ff)

        return self.add_norm(ff_out, ff_in)

#Layer to add and normalize two values
class AddNorm(Layer):
    def __init__(self):
        super().__init__()
        self.norm = LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, y):
        sum = self.add([x, y])
        return self.norm(sum)

#Transforms tokenized input into embedded feature vectors with positional encoding
class Preprocessing(Layer):
    def __init__(self, vocab_size, embedding_dim, max_input_size):
        super().__init__()
        self.embed = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = positional_encoding(max_input_size, embedding_dim)
        

    #Takes tokenized input of size batch_size x N. Outputs tensor of size batch_size x N x embedding_dim
    def call(self, x): 
        input_length = tf.shape(x)[1]
        x = self.embed(x)
        position_vectors = self.positional_encoding[tf.newaxis, :input_length, :]

        return tf.math.add(x, position_vectors)

#Generates positional encoding matrix given length and depth
def positional_encoding(length, depth):
        depth = int(depth/2)

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        rads = positions / (10000**depths)

        pos_encoding = np.concatenate(
            [np.sin(rads), np.cos(rads)],
            axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)