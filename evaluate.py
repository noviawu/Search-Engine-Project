"""
same usage as evaluation.py, this file is used for evaluating the different
queries and retrieval methods used
adopted from Curtis's evaluate_old.py
author: Novia
"""
import argparse

from elasticsearch_dsl.connections import connections
from fp import bm25_documents, embedding_documents
from metrics import ndcg, average_precision


def form_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        required=True,
        type=str,
        help="name of the ES index",
    )
    parser.add_argument(
        "--topic_id",
        required=True,
        type=str,
        help="topic number to access",
    )
    parser.add_argument(
        '--query_type',
        choices=['title', 'description', 'narration'],
        required=True,
        type=str,
        help='the query to use'
    )
    parser.add_argument(
        '--vector_name',
        choices=['sbert_vector', 'ft_vector'],
        required=False,
        type=str,
        help='which word embeddings vector to use'
    )
    parser.add_argument(
        '--top_k',
        required=True,
        type=unsigned_int,
        help='number of hits to return (greater than or equal to zero)'
    )
    parser.add_argument(
        '--analyzer',
        choices=['default', 'n_gram', 'whitespace'],
        required=True,
        type=str,
        help='the analyzer to use'
    )
    return parser


def unsigned_int(x):
    """
    new "type": unsigned int
    :author: Curtis
    :param x: variable to cast to unsigned int
    :return: int form if x is a non-zero int
    :raise argparse.ArgumentTypeError: if x is not a non-zero int
    """
    if int(x) < 0:
        raise argparse.ArgumentTypeError('Must be a number greater than or equal to zero.')
    return int(x)


def print_result_rprecision(response, top_k):
    """
    helper method used to print the r precision and the f score
    eventually decided to not use this evaluation matrix
    because r precision and f socre do not take into consideration the order of ranked documents
    :author: Novia
    :param response: ElasticSearch from `search.execute()`
    :param top_k: number of documents to retrieve
    """
    true_pos = 0
    total_relevant = 135
    for hit in response:
        if hit.annotation:
            if hit.annotation[-1] == '2' or hit.annotation[-1] == '1':
                true_pos += 1
    print("true pos", true_pos)
    p = true_pos / top_k
    r = true_pos / total_relevant
    print("r-precision is ", r)
    print("f score is", (2 * p * r) / (p + r))


def get_search(query, analyzer, top_k, vector_name):
    """
    find the relevant docs or irrelevant docs (depending on query) using ES
    :author: Novia
    :param query: query to query on
    :param analyzer: which analyzer to use
    :param top_k: number of documents to retrieve
    :param vector_name: which vector to use for reranking
    :return: results of the search
    """
    bm_search = bm25_documents(query, analyzer, top_k)  # a search object to match docs
    if vector_name:
        ranker = 'sbert' if vector_name == 'sbert_vector' else 'fasttext'
        new_search = embedding_documents(query, bm_search, ranker, top_k)
        results = new_search.execute()
    else:  # bm25
        results = bm_search.execute()
    return results


def get_final_scores(r1, r2, topic):
    """
    uses set difference on r1 and r2 and then calculates NDCG and average precision using what is left
    :author: Novia
    :param r1: search result with relevant documents
    :param r2: search results with irrelevant documents
    :param topic: the topic number (for our case, 816)
    """
    # add all the irrelevant docs into a set
    s = set()
    for hit2 in r2:
        s.add(hit2.doc_id)
    relevance_list = []
    # getting the ndcg for filtered docs
    i = 1
    for hit in r1:
        if hit.doc_id not in s:
            annotation: str = hit.annotation
            if annotation:
                if annotation[:3] == str(topic):
                    score = int(annotation[-1])
                    print(i, annotation, hit.title, sep="\t")
                    i += 1
                else:
                    score = 0
            else:
                score = 0
            relevance_list.append(score)
        else:
            print("irrelevant: ", hit.annotation, hit.title, sep="\t")
    print(relevance_list)
    print("ndcg:", ndcg(relevance_list))
    print("ave precision:", average_precision(relevance_list))


# adopted by Novia from Curtis's code
def main():
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")
    parser = form_parser()
    args = parser.parse_args()

    # modified the description and narration of the queries manually
    query_list = ["Federal Minimum Wage Increase",
                  "actions and reactions of President or Congress to increase U.S. federal minimum wage",
                  "advocacy or actions (or lack thereof) by the President or Congress to increase the U.S. federal minimum wage,government contract workers."]
    query = query_list[0] if args.query_type == 'title' else query_list[1] if args.query_type == 'description' else query_list[2]

    if args.analyzer == 'n_gram':
        analyzer = 'n_gram'
    elif args.analyzer == 'whitespace':
        analyzer = 'whitespace'
    else:
        analyzer = 'default'
    top_k = int(args.top_k)

    # find relevant matching docs
    rel_result = get_search(query, analyzer, top_k, args.vector_name)

    # find irrelevant docs
    irrelevant_query = "Analyses discussions pros and cons U.S. federal minimum wage increase by talking heads"
    irrelevant_result = get_search(irrelevant_query, analyzer, top_k, args.vector_name)

    # get relevance
    get_final_scores(rel_result, irrelevant_result, args.topic_id)


if __name__ == '__main__':
    main()
