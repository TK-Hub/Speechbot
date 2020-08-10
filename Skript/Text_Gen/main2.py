#==================================================================================================
#                       Main Script Speechbot
#                       Author: T.K.
#                       Created: 28.04.2020
#==================================================================================================
import tensorflow as tf
import numpy as np
import os
import time


#==================================================================================================
# 1. Data Preparation
#==================================================================================================

text = open("full_speech.txt", 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(vocab)

# Indexing the individual words
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

print(char2idx)
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# Divide the text into input pieces. // is a division, rounded down to a whole number. The target
# sentence will have the same length, shifted one to the right. That is why we divide the text into chunks
# of x+1. 

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Convert into Tensorflow format? This takes the int-vales from the text created above.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

# The batch method converts the individual letters into chunks of the desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Create the dataset
dataset = sequences.map(split_input_target)

# Exemplify what the RNN tries to predict (Indexes of the next letter, based on the previous one)
for input_example, target_example in dataset.take(1):
    print('Input data:', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ',  repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Create training batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#print(dataset)


#==================================================================================================
# 2. Building the Model
#==================================================================================================

# Length of the vocabulary in chars
vocab_size = len(vocab)

# Embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
    batch_size = BATCH_SIZE
)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    #print(example_batch_predictions.shape,  "# (batch_size, sequence_length, vocab_size)")

print(model.summary())

# Trying it for the first example in the batch
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=1).numpy()

#print(sampled_indices)

# Decoded, that means:
#print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
#print()
#print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


#==================================================================================================
# 3. Training the Model
#==================================================================================================

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
#print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
#print("scalar_loss:      ", example_batch_loss.numpy().mean())

# Compiling the Model, specifying the Adam-Optimizer and the loss-function as defined above
model.compile(optimizer='adam', loss=loss)

# Configuring Checkpoints
checkpoint_dir = './training_checkpoints_t'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS=35
#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


#==================================================================================================
# 4. Applying the Model (Text Generation)
#==================================================================================================

# Print last checkpoint
print(tf.train.latest_checkpoint(checkpoint_dir))

# Re-initializing the model and filling in the weights from the checkpoint.
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.75

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"Trump: "))