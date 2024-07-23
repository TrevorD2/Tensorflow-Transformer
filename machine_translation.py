import tensorflow as tf
from transformer import Transformer
import re
import io
import os

#Download and load the dataset
path_to_zip = tf.keras.utils.get_file(
    'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip', extract=True)
path_to_file = os.path.join(os.path.dirname(path_to_zip), "fra.txt")

#Cleans sentence and adds start and end tokens
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

#Takes the first num_examples datapoints and converts them to a tuple of (input_language, target_language)
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

#Tokenizes and pads input
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    
    #Add '<unk>' token if not already in the tokenizer
    if '<unk>' not in lang_tokenizer.word_index:
        lang_tokenizer.word_index['<unk>'] = len(lang_tokenizer.word_index) + 1
    
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    
    return tensor, lang_tokenizer

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

#Dataset and training variables
num_examples = 100000
train_test_split = 0.8
BATCH_SIZE = 64

input_lang, target_lang = create_dataset(path_to_file, num_examples)

input_tensor, inp_lang_tokenizer = tokenize(input_lang)
target_tensor, targ_lang_tokenizer = tokenize(target_lang)

max_length_inp, max_length_targ = input_tensor.shape[1], target_tensor.shape[1]

BUFFER_SIZE = len(input_tensor)

#Create dec_inputs and dec_targets from target_tensor
dec_inputs = target_tensor[:, :-1]  #All but the last token
dec_targets = target_tensor[:, 1:]  #All but the first token

#Create the training and testing datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((input_tensor[:int(len(input_tensor)*train_test_split)], dec_inputs[:int(len(dec_inputs)*train_test_split)]), dec_targets[:int(len(dec_targets)*train_test_split)]))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

validation_dataset = tf.data.Dataset.from_tensor_slices(((input_tensor[int(len(input_tensor)*train_test_split):], dec_inputs[int(len(dec_inputs)*train_test_split):]), dec_targets[int(len(dec_targets)*train_test_split):]))
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)

#Learning rate scheduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



#Transformer parameters
context_vocab_size = len(inp_lang_tokenizer.word_index) + 1
target_vocab_size = len(targ_lang_tokenizer.word_index) + 1
max_input_size = max(max_length_inp, max_length_targ)

#Hyperparameters: modify to fit performace needs
model_dim = 128
stack_height = 4
heads = 8
key_dim = 32
ff_dim = 512
dropout = 0.1

#Create transformer
transformer = Transformer(context_vocab_size, 
                          model_dim, 
                          max_input_size, 
                          stack_height, 
                          heads, 
                          key_dim, 
                          ff_dim, 
                          dropout, 
                          target_vocab_size)

#Create optimizer
learning_rate = CustomSchedule(model_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)

#Compile and train the model
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_dataset,
                epochs=20,
                validation_data=validation_dataset)