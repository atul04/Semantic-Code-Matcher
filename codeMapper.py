# @Author: Atul Sahay <atul>
# @Date:   2020-04-05T16:18:05+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-04-07T07:23:55+05:30
import dill as dpickle
import numpy as np
from ktext.preprocess import processor
import argparse
from seq2seq_utils import Seq2Seq_Inference
from keras.models import Model, load_model
import pandas as pd
import logging
from general_utils import get_step2_prerequisite_files, read_training_files
from keras.utils import get_file
from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='For the inference of the model')
parser.add_argument('-d','--download',type=str2bool,nargs='?',const=True,default=True,
                   help='To download the model or not')
parser.add_argument('-I','--input_file', type=str,required=True,
                   help='Input file which contains the code')
parser.add_argument('-O','--output_file', type=str,required=False,
                   help='Destination File which will contain the docstring')

args = vars(parser.parse_args())

# print(args['download'])
# print(args['input_file'])
# print(args['output_file'])


# if download = True, it will download the model from the net.
# or else it will use the saved model
filename = ""
if(args['download']):

    import wget

    url = 'https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_func_sum_v9_.epoch16-val2.55276.hdf5'
    filename = wget.download(url, out='./data/seq2seq/')

else:
    filename = 'data/seq2seq/code_summary_seq2seq_model.h5'



seq2seq_Model = load_model(filename)


loc = ""
# Load encoder (code) pre-processor from url
if(args['download']):
    loc = get_file(fname='py_code_proc_v2.dpkl',
               origin='https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_code_proc_v2.dpkl')
else:
    loc="data/seq2seq/py_code_proc_v2.dpkl"
num_encoder_tokens, enc_pp = load_text_processor(loc)

loc = ""
# Load encoder (code) pre-processor from url
if(args['download']):
# Load decoder (docstrings/comments) pre-processor from url
    loc = get_file(fname='py_comment_proc_v2.dpkl',
               origin='https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_comment_proc_v2.dpkl')

else:
    loc="data/seq2seq/py_comment_proc_v2.dpkl"
num_decoder_tokens, dec_pp = load_text_processor(loc)

from seq2seq_utils import Seq2Seq_Inference
import pandas as pd

seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                 decoder_preprocessor=dec_pp,
                                 seq2seq_model=seq2seq_Model)
print("Reading a File")
with open(args['input_file'], 'r') as f:
    t_enc = f.readlines()

out_doc = []
print("Prediction going on")
for codeInstance in t_enc:
    outDocInst=seq2seq_inf.predict(codeInstance)[1]
    out_doc.append([codeInstance,outDocInst])

print("writing the file")
import csv
with open(args['output_file'], 'w') as f:
    csv_out=csv.writer(f)
    csv_out.writerow(['Code','Docstring'])
    for item in out_doc:
        csv_out.writerow(item)
print("Completed")
