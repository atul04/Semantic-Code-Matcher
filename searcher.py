# @Author: Atul Sahay <atul>
# @Date:   2020-04-07T11:07:06+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-04-07T11:24:12+05:30

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
parser.add_argument('-direct','--direct',type=str2bool,nargs='?',const=True,default=False,
                   help='Entering direct into the search mode')
# parser.add_argument('-I','--input_file', type=str,required=True,
#                    help='Input file which contains the code')
# parser.add_argument('-O','--output_filer, type=str,required=False,
#                    help='Destination File which will contain the docstring')

args = vars(parser.parse_args())
print(args['direct'])

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib
from lang_model_utils import load_lm_vocab, Query2Emb
from general_utils import create_nmslib_search_index

input_path = Path('./data/processed_data/')
code2emb_path = Path('./data/code2emb/')
output_path = Path('./data/search')
output_path.mkdir(exist_ok=True)

if(not args['direct']):
    print("reading the code and urls")
    # read file of urls
    url_df = pd.read_csv(input_path/'without_docstrings.lineage',header=None, names=['url'])
    # url_df = url_df.iloc[1:]
    # read original code
    code_df = pd.read_json(input_path/'without_docstrings_original_function.json.gz')
    code_df.columns = ['code']
    # print(code_df.shape)
    # make sure these files have same number of rows
    assert code_df.shape[0] == url_df.shape[0]

    # collect these two together into a dataframe
    ref_df = pd.concat([url_df, code_df], axis = 1).reset_index(drop=True)
    print(ref_df.head())

    print("Creating the Search Index For Vectorized Code")

    nodoc_vecs = np.load(code2emb_path/'nodoc_vecs.npy')
    assert nodoc_vecs.shape[0] == ref_df.shape[0]
    search_index = create_nmslib_search_index(nodoc_vecs)
    print("Saving the search Index")
    search_index.saveIndex('./data/search/search_index.nmslib')

print("Building the minimal search Index")
lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch',
                        map_location=lambda storage, loc: storage)

vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

search_index = nmslib.init(method='hnsw', space='cosinesimil')
search_index.loadIndex('./data/search/search_index.nmslib')
print("Search Index loaded for the vectorised code")

print("Activating the searcher")
class search_engine:
    """Organizes all the necessary elements we need to make a search engine."""
    def __init__(self,
                 nmslib_index,
                 ref_df,
                 query2emb_func):
        """
        Parameters
        ==========
        nmslib_index : nmslib object
            This is pre-computed search index.
        ref_df : pandas.DataFrame
            This dataframe contains meta-data for search results,
            must contain the columns 'code' and 'url'.
        query2emb_func : callable
            This is a function that takes as input a string and returns a vector
            that is in the same vector space as what is loaded into the search index.

        """
        assert 'url' in ref_df.columns
        assert 'code' in ref_df.columns

        self.search_index = nmslib_index
        self.ref_df = ref_df
        self.query2emb_func = query2emb_func

    def search(self, str_search, k=2):
        """
        Prints the code that are the nearest neighbors (by cosine distance)
        to the search query.

        Parameters
        ==========
        str_search : str
            a search query.  Ex: "read data into pandas dataframe"
        k : int
            the number of nearest neighbors to return.  Defaults to 2.

        """
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)

        for idx, dist in zip(idxs, dists):
            code = self.ref_df.iloc[idx].code
            url = self.ref_df.iloc[idx].url
            print(f'cosine dist:{dist:.4f}  url: {url}\n---------------\n')
            print(code)

# read file of urls
url_df = pd.read_csv(input_path/'without_docstrings.lineage',header=None, names=['url'])
# read original code
code_df = pd.read_json(input_path/'without_docstrings_original_function.json.gz')
code_df.columns = ['code']
# make sure these files have same number of rows
assert code_df.shape[0] == url_df.shape[0]
# collect these two together into a dataframe
ref_df = pd.concat([url_df, code_df], axis = 1).reset_index(drop=True)


se = search_engine(nmslib_index=search_index,
                   ref_df=ref_df,
                   query2emb_func=q2emb.emb_mean)

t = input("Search Code: ")
while(len(t)):
    se.search(t,5)
    t = input("Search Code: ")
