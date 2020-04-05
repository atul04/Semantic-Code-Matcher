# @Author: Atul Sahay <atul>
# @Date:   2020-04-05T11:41:42+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-04-05T16:21:07+05:30

# For the Setting of the use_cache values
'''
if use_cache = True, data will be downloaded where possible instead of re-computing.
However, it is highly recommended that you set use_cache = False
'''

use_cache = False

# # Optional: you can set what GPU you want to use.
# # Useful if you want to run concurrent experiments at the same time on different GPUs.
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Setting the ouput directory
from pathlib import Path
from general_utils import get_step2_prerequisite_files, read_training_files
from keras.utils import get_file
OUTPUT_PATH = Path('./data/seq2seq/')
OUTPUT_PATH.mkdir(exist_ok=True)

print("Starting to read from the file")
# Read Text From File
if use_cache:
    get_step2_prerequisite_files(output_directory = './data/processed_data')

# you want to supply the directory where the files are from step 1.
train_code, holdout_code, train_comment, holdout_comment = read_training_files('./data/processed_data/')
print("Done with the read")

# code and comment files should be of the same length.
assert len(train_code) == len(train_comment)
assert len(holdout_code) == len(holdout_comment)


# Tokenize Text
print("Starting Tokenizing the text")

from ktext.preprocess import processor

if not use_cache:
    code_proc = processor(heuristic_pct_padding=.7, keep_n=20000)
    t_code = code_proc.fit_transform(train_code)

    comment_proc = processor(append_indicators=True, heuristic_pct_padding=.7, keep_n=14000, padding ='post')
    t_comment = comment_proc.fit_transform(train_comment)

elif use_cache:
    logging.warning('Not fitting transform function because use_cache=True')
print("Done with the tokenization")

# Saving the tokenizing
print("Saving the tokenizised text")
import dill as dpickle
import numpy as np

if not use_cache:
    # Save the preprocessor
    with open(OUTPUT_PATH/'py_code_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(code_proc, f)

    with open(OUTPUT_PATH/'py_comment_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(comment_proc, f)

    # Save the processed data
    np.save(OUTPUT_PATH/'py_t_code_vecs_v2.npy', t_code)
    np.save(OUTPUT_PATH/'py_t_comment_vecs_v2.npy', t_comment)
print("Done with saving of the tokenizised text")

from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor

# Encoding of the input and output texts
print("Encoding the input and output text")
encoder_input_data, encoder_seq_len = load_encoder_inputs(OUTPUT_PATH/'py_t_code_vecs_v2.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs(OUTPUT_PATH/'py_t_comment_vecs_v2.npy')
num_encoder_tokens, enc_pp = load_text_processor(OUTPUT_PATH/'py_code_proc_v2.dpkl')
num_decoder_tokens, dec_pp = load_text_processor(OUTPUT_PATH/'py_comment_proc_v2.dpkl')

# Build Seq2Seq Model For Summarizing Code
print("Building the model")
from seq2seq_utils import build_seq2seq_model
seq2seq_Model = build_seq2seq_model(word_emb_dim=800,
                                    hidden_state_dim=1000,
                                    encoder_seq_len=encoder_seq_len,
                                    num_encoder_tokens=num_encoder_tokens,
                                    num_decoder_tokens=num_decoder_tokens)
print(seq2seq_Model.summary())

# Train Seq2Seq Model
print("Starting the trainig")
from keras.models import Model, load_model
import pandas as pd
import logging

if not use_cache:

    from keras.callbacks import CSVLogger, ModelCheckpoint
    import numpy as np
    from keras import optimizers

    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.00005), loss='sparse_categorical_crossentropy')

    script_name_base = 'py_func_sum_v9_'
    csv_logger = CSVLogger('{:}.log'.format(script_name_base))

    model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                       save_best_only=True)

    batch_size = 1100
    epochs = 16
    history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.12, callbacks=[csv_logger, model_checkpoint])

from seq2seq_utils import Seq2Seq_Inference
import pandas as pd

seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                 decoder_preprocessor=dec_pp,
                                 seq2seq_model=seq2seq_Model)

demo_testdf = pd.DataFrame({'code':holdout_code, 'comment':holdout_comment, 'ref':''})

print("Starting the evaluation on the holdout set")
# This will return a BLEU Score
seq2seq_inf.evaluate_model(input_strings=holdout_code,
                           output_strings=holdout_comment,
                           max_len=None)

seq2seq_Model.save(OUTPUT_PATH/'code_summary_seq2seq_model.h5')
Print("Model saved in",OUTPUT_PATH)
