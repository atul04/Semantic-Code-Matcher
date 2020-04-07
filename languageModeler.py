# @Author: Atul Sahay <atul>
# @Date:   2020-04-07T07:22:57+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-04-07T08:35:34+05:30

# # Optional: you can set what GPU you want to use in a notebook like this.
# # Useful if you want to run concurrent experiments at the same time on different GPUs.
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse

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
parser.add_argument('-eval','--eval',type=str2bool,nargs='?',const=True,default=False,
                   help='Entering into the evaluation mode')
# parser.add_argument('-I','--input_file', type=str,required=True,
#                    help='Input file which contains the code')
# parser.add_argument('-O','--output_filer, type=str,required=False,
#                    help='Destination File which will contain the docstring')

args = vars(parser.parse_args())
print(args['eval'])
# exit()
'''
If use_cache = True, data will be downloaded where possible instead of re-computing.
However, it is highly recommended that you set use_cache = False
'''
if(not args['eval']):
    use_cache = False


    ## Download pre-processed data, If not already done.........
    from general_utils import get_step2_prerequisite_files

    if use_cache:
        get_step2_prerequisite_files(output_directory = './data/processed_data')
    ######################### Downloading the output directory######################


    # Build Language Model From Docstrings

    '''
    The goal is to build a language model using the docstrings,
    and use that language model to generate an embedding for each docstring.
    '''

    import torch,cv2
    from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
    from general_utils import save_file_pickle, load_file_pickle
    import logging
    from pathlib import Path
    from fastai.text import * # created a symbolic link with the code : ln -s fastai/old/fastai fastai

    source_path = Path('./data/processed_data/')

    print("Reading the files")
    with open(source_path/'train.docstring', 'r') as f:
        trn_raw = f.readlines()

    with open(source_path/'valid.docstring', 'r') as f:
        val_raw = f.readlines()

    with open(source_path/'test.docstring', 'r') as f:
        test_raw = f.readlines()
    print("Done with the reading")
    '''
    Note: its important that fastai soft link does exists
    '''
    import fastai.text


    # Pre-process data for language model

    '''
    We will use the class build_lm_vocab to prepare our data for the language model

    '''

    vocab = lm_vocab(max_vocab=50000,
                     min_freq=10)
    print("Start building the vocabulary")
    # fit the transform on the training data, then transform
    trn_flat_idx = vocab.fit_transform_flattened(trn_raw)

    # apply transform to validation data
    val_flat_idx = vocab.transform_flattened(val_raw)
    print("Done with the vocaulary build task")

    # Need to check whether the data/lang_model exists or not
    from pathlib import Path
    #where you will save artifacts from this step
    OUTPUT_PATH = Path('./data/lang_model/')
    OUTPUT_PATH.mkdir(exist_ok=True)

    print("saving the vocab in data/lang_model")
    if not use_cache:
        vocab.save('./data/lang_model/vocab_v2.cls')
        save_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2', trn_flat_idx)
        save_file_pickle('./data/lang_model/val_flat_idx_list.pkl_v2', val_flat_idx)


    ################# Train Fast.AI Language Model

    '''
    This model will read in files that were created and train a fast.ai language model.
    This model learns to predict the next word in the sentence using fast.ai's implementation of AWD LSTM.

    The goal of training this model is to build a general purpose feature extractor for text that can
    be used in downstream models. In this case, we will utilize this model to produce embeddings for function docstrings.

    '''
    vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
    trn_flat_idx = load_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2')
    val_flat_idx = load_file_pickle('./data/lang_model/val_flat_idx_list.pkl_v2')



    print("start training the model")
    if not use_cache:
        fastai_learner, lang_model = train_lang_model(model_path = './data/lang_model_weights_v2',
                                                      trn_indexed = trn_flat_idx,
                                                      val_indexed = val_flat_idx,
                                                      vocab_size = vocab.vocab_size,
                                                      lr=3e-3,
                                                      em_sz= 500,
                                                      nh= 500,
                                                      bptt=20,
                                                      cycle_len=1,
                                                      n_cycle=3,
                                                      cycle_mult=2,
                                                      bs = 300,
                                                      wd = 1e-6)

    elif use_cache:
        logging.warning('Not re-training language model because use_cache=True')
    if not use_cache:
        fastai_learner.fit(1e-3, 3, wds=1e-6, cycle_len=2)
    if not use_cache:
        fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=2)
    if not use_cache:
        fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=10)

    print("Saving the trained model in data/lang_model/")
    # Saving the model
    if not use_cache:
        fastai_learner.save('lang_model_learner_v2.fai')
        lang_model_new = fastai_learner.model.eval()
        torch.save(lang_model_new, './data/lang_model/lang_model_gpu_v2.torch')
        torch.save(lang_model_new.cpu(), './data/lang_model/lang_model_cpu_v2.torch')

    ################## Load Model and Encode All Docstrings
    '''
    Now that we have trained the language model, the next step is to use the language model
    to encode all of the docstrings into a vector.
    '''

    print("vectorising the whole docstring for the later use")
    from lang_model_utils import load_lm_vocab
    vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
    idx_docs = vocab.transform(trn_raw + val_raw, max_seq_len=30, padding=False)
    lang_model = torch.load('./data/lang_model/lang_model_gpu_v2.torch',
                            map_location=lambda storage, loc: storage)
    lang_model.eval()

    '''
    Extracts the embedding for each docstring one at a time.

    '''

    def list2arr(l):
        "Convert list into pytorch Variable."
        return V(np.expand_dims(np.array(l), -1)).cpu()

    def make_prediction_from_list(model, l):
        """
        Encode a list of integers that represent a sequence of tokens.  The
        purpose is to encode a sentence or phrase.

        Parameters
        -----------
        model : fastai language model
        l : list
            list of integers, representing a sequence of tokens that you want to encode

        """
        arr = list2arr(l)# turn list into pytorch Variable with bs=1
        model.reset()  # language model is stateful, so you must reset upon each prediction
        hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer

        #return avg-pooling, max-pooling, and last hidden state
        return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


    def get_embeddings(lm_model, list_list_int):
        """
        Vectorize a list of sequences List[List[int]] using a fast.ai language model.

        Paramters
        ---------
        lm_model : fastai language model
        list_list_int : List[List[int]]
            A list of sequences to encode

        Returns
        -------
        tuple: (avg, mean, last)
            A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
        """
        n_rows = len(list_list_int)
        n_dim = lm_model[0].nhid
        avgarr = np.empty((n_rows, n_dim))
        maxarr = np.empty((n_rows, n_dim))
        lastarr = np.empty((n_rows, n_dim))

        for i in tqdm_notebook(range(len(list_list_int))):
            avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
            avgarr[i,:] = avg_.data.numpy()
            maxarr[i,:] = max_.data.numpy()
            lastarr[i,:] = last_.data.numpy()

        return avgarr, maxarr, lastarr

    # Takes around 4 hour
    print("ectorize a list of sequences List[List[int]] using a fast.ai language model.")
    avg_hs, max_hs, last_hs = get_embeddings(lang_model, idx_docs)

    print("for test set")
    idx_docs_test = vocab.transform(test_raw, max_seq_len=30, padding=False)
    avg_hs_test, max_hs_test, last_hs_test = get_embeddings(lang_model, idx_docs_test)

    print('Saving Language Model Embeddings For Docstrings')
    savepath = Path('./data/lang_model_emb/')
    savepath.mkdir(exist_ok=True)
    np.save(savepath/'avg_emb_dim500_v2.npy', avg_hs)
    np.save(savepath/'max_emb_dim500_v2.npy', max_hs)
    np.save(savepath/'last_emb_dim500_v2.npy', last_hs)

    np.save(savepath/'avg_emb_dim500_test_v2.npy', avg_hs_test)
    np.save(savepath/'max_emb_dim500_test_v2.npy', max_hs_test)
    np.save(savepath/'last_emb_dim500_test_v2.npy', last_hs_test)


#### For the manual evaluation of the language model

print("Starting the manual evaluation, It will take couple of minutes")
from general_utils import create_nmslib_search_index
import nmslib
from lang_model_utils import Query2Emb
from pathlib import Path
import numpy as np
from lang_model_utils import load_lm_vocab
import torch




source_path = Path('./data/processed_data/')

with open(source_path/'train.docstring', 'r') as f:
    trn_raw = f.readlines()

with open(source_path/'valid.docstring', 'r') as f:
    val_raw = f.readlines()

with open(source_path/'test.docstring', 'r') as f:
    test_raw = f.readlines()



# Load matrix of vectors
loadpath = Path('./data/lang_model_emb/')
avg_emb_dim500 = np.load(loadpath/'avg_emb_dim500_test_v2.npy')

print("Building the search index")
# Build search index (takes about an hour on a p3.8xlarge)
dim500_avg_searchindex = create_nmslib_search_index(avg_emb_dim500)

print("Saving the search index for the later use")
# save search index
dim500_avg_searchindex.saveIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')

dim500_avg_searchindex = nmslib.init(method='hnsw', space='cosinesimil')
dim500_avg_searchindex.loadIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')

lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch')
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')

q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

class search_engine:
    def __init__(self,
                 nmslib_index,
                 ref_data,
                 query2emb_func):

        self.search_index = nmslib_index
        self.data = ref_data
        self.query2emb_func = query2emb_func

    def search(self, str_search, k=3):
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)

        for idx, dist in zip(idxs, dists):
            print(f'cosine dist:{dist:.4f}\n---------------\n', self.data[idx])

print("Starting the search engine")
se = search_engine(nmslib_index=dim500_avg_searchindex,
                   ref_data = test_raw,
                   query2emb_func = q2emb.emb_mean)

import logging
logging.getLogger().setLevel(logging.ERROR)

t=input("Search text: ")
while(len(t)):
    se.search(t,5)
    t = input("Search text: ")
