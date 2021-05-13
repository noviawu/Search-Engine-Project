"""
This class counts the term frequency and the document frequency of each terms in all docs
and pickle them for later use
we want to "filter" the terms with low tf-idf for a better fast-text embedding
adopted from Jingxuan's code by Novia
"""

import collections
import pickle
from tqdm import tqdm
from utils import load_clean_wapo_with_embedding
from embedding_service.text_processing import TextProcessing

if __name__ == '__main__':
    text_processor = TextProcessing.from_nltk()

    # a counter of document frequency, keeps track of all the terms and their df
    # {'a':2,'b':3 ...}
    df_counter = collections.Counter()

    # a dictionary that keeps track of each doc's term frequency, number int is the key and tf coutner is value
    # {0: {hi:2,bye:3,happy:9..}, 1: {bye:2,happy:1..} ...}
    doc_tf_dict = {}

    # change pa5_data -> fp_data
    wapo_file = "pa5_data/subset_wapo_50k_sbert_ft_filtered.jl"
    for i, doc in tqdm(enumerate(load_clean_wapo_with_embedding(wapo_file))):
        # print(i) # count of docs from 0 to xxxxx
        # print(doc) # the dictionary object of one doc, like {'title':'xxx', 'content_str':'ddd'...}
        title = doc.get("title", "") if doc.get("title") else ""
        content = doc.get("content_str", "") if doc.get("content_str") else ""

        # get all valid tokens from the title and content
        tokens = text_processor.get_valid_tokens(title, content, use_stemmer=True)

        # a counter object that has tokens mapped to their counts, this is the raw term frequency
        tf_dict = collections.Counter(tokens)

        # appending to the dictionary of tf counter
        doc_tf_dict[i] = tf_dict

        # keep counting the df for each term
        df_counter.update(tf_dict.keys())

    print(len(df_counter))  # number of terms in the corpus
    print(len(doc_tf_dict))  # number of documents in the corpus
    file1 = open("df_counter.pkl", "wb")
    file2 = open("doc_tf_dict.pkl", "wb")
    pickle.dump(df_counter, file1)
    pickle.dump(df_counter, file2)
    file1.close()
    file2.close()
