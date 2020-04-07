# @Author: Atul Sahay <atul>
# @Date:   2020-04-07T10:01:50+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-04-07T10:15:52+05:30

# # Optional: you can set what GPU you want to use in a notebook like this.
# # Useful if you want to run concurrent experiments at the same time on different GPUs.
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

from pathlib import Path
import numpy as np
from seq2seq_utils import extract_encoder_model, load_encoder_inputs
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda

from keras.models import load_model, Model
from seq2seq_utils import load_text_processor

#where you will save artifacts from this step
OUTPUT_PATH = Path('./data/code2emb/')
OUTPUT_PATH.mkdir(exist_ok=True)

# These are where the artifacts are stored from steps 2 and 3, respectively.
seq2seq_path = Path('./data/seq2seq/')
langemb_path = Path('./data/lang_model_emb/')

# set seeds
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


print("Loading seq2seq model and extracting the encoder")
# load the pre-processed data for the encoder (we don't care about the decoder in this step)
encoder_input_data, doc_length = load_encoder_inputs(seq2seq_path/'py_t_code_vecs_v2.npy')
seq2seq_Model = load_model(str(seq2seq_path/'code_summary_seq2seq_model.h5'))


# Extract Encoder from seq2seq model
encoder_model = extract_encoder_model(seq2seq_Model)
# Get a summary of the encoder and its layers
print(encoder_model.summary())



print("Freezing the layers")
# Freeze Encoder Model
for l in encoder_model.layers:
    l.trainable = False
    print(l, l.trainable)


print("loading the language embedders")
# Load Fitlam Embeddings
fastailm_emb = np.load(langemb_path/'avg_emb_dim500_v2.npy')

# check that the encoder inputs have the same number of rows as the docstring embeddings
assert encoder_input_data.shape[0] == fastailm_emb.shape[0]

print(fastailm_emb.shape)

print("Constructing code2emb Model Architecture")
#### Encoder Model ####
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')
enc_out = encoder_model(encoder_inputs)

# first dense layer with batch norm
x = Dense(500, activation='relu')(enc_out)
x = BatchNormalization(name='bn-1')(x)
out = Dense(500)(x)
code2emb_model = Model([encoder_inputs], out)
print(code2emb_model.summary())


print("Starting the training")
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import optimizers

code2emb_model.compile(optimizer=optimizers.Nadam(lr=0.002), loss='cosine_proximity')
script_name_base = 'code2emb_model_'
csv_logger = CSVLogger('{:}.log'.format(script_name_base))
model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                   save_best_only=True)

batch_size = 20000
epochs = 15
history = code2emb_model.fit([encoder_input_data], fastailm_emb,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.12, callbacks=[csv_logger, model_checkpoint])

print("Unfreeze all Layers of Model and Resume Training")

for l in code2emb_model.layers:
    l.trainable = True
    print(l, l.trainable)
code2emb_model.compile(optimizer=optimizers.Nadam(lr=0.0001), loss='cosine_proximity')
script_name_base = 'code2emb_model_unfreeze_'
csv_logger = CSVLogger('{:}.log'.format(script_name_base))
model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                   save_best_only=True)

batch_size = 2000
epochs = 20
history = code2emb_model.fit([encoder_input_data], fastailm_emb,
          batch_size=batch_size,
          epochs=epochs,
          initial_epoch=16,
          validation_split=0.12, callbacks=[csv_logger, model_checkpoint])
print("saving the model")
code2emb_model.save(OUTPUT_PATH/'code2emb_model.hdf5')

print("Vectorizing all of the code without docstrings\n for the manual inspection at the later stage")
from keras.models import load_model
from pathlib import Path
import numpy as np
from seq2seq_utils import load_text_processor
code2emb_path = Path('./data/code2emb/')
seq2seq_path = Path('./data/seq2seq/')
data_path = Path('./data/processed_data/')

code2emb_model = load_model(str(code2emb_path/'code2emb_model.hdf5'))
num_encoder_tokens, enc_pp = load_text_processor(str(seq2seq_path/'py_code_proc_v2.dpkl'))

with open(data_path/'without_docstrings.function', 'r') as f:
    no_docstring_funcs = f.readlines()

print("Preprocessing code without docstrings for input into code2emb model")
encinp = enc_pp.transform_parallel(no_docstring_funcs)
np.save(code2emb_path/'nodoc_encinp.npy', encinp)

print("extracting the code vectors")
from keras.models import load_model
from pathlib import Path
import numpy as np
code2emb_path = Path('./data/code2emb/')
encinp = np.load(code2emb_path/'nodoc_encinp.npy')
code2emb_model = load_model(str(code2emb_path/'code2emb_model.hdf5'))

print("starting the vectorising the docstring [heavy task]")
nodoc_vecs = code2emb_model.predict(encinp, batch_size=20000)

print("Saving the vector file")
np.save(code2emb_path/'nodoc_vecs.npy', nodoc_vecs)
print("Task completed")
