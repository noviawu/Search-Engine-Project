"""
# SAMPLE QUERIES!
# use title from topic 321 as the query; search over the custom_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type title -u --top_k 20

# use narration from topic 321 as the query; search over the content field from index "wapo_docs_50k" based on sentence BERT embedding and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type narration --vector_name sbert_vector --top_k 20
"""
import argparse
from pathlib import Path
from utils import parse_wapo_topics

from elasticsearch_dsl.connections import connections
from hw5 import bm25_documents, embedding_documents
from metrics import ndcg


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
        required=True,
        type=str,
        help='one of [title, description, narration]'
    )
    parser.add_argument(
        '--vector_name',
        required=False,
        type=str,
        help='one of [sbert_vector, ft_vector]'
    )
    parser.add_argument(
        '--top_k',
        required=True,
        type=int,
        help='number of hits to return'
    )
    parser.add_argument(
        '-u',
        action='store_true',
        help='include this argument if using the custom analyzer (otherwise default)'
    )
    return parser


def main():
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    title = 0
    description = 1
    narration = 2

    parser = form_parser()
    args = parser.parse_args()

    topics = parse_wapo_topics(f'{Path("pa5_data").joinpath("topics2018.xml")}')

    idx = title if args.query_type == 'title' else narration if args.query_type == 'narration' else description
    query = topics[args.topic_id][idx]
    analyzer = 'custom' if args.u else 'default'
    top_k = int(args.top_k)

    bm_search = bm25_documents(query, analyzer, top_k)

    if args.vector_name:
        ranker = 'sbert' if args.vector_name == 'sbert_vector' else 'fasttext'
        new_search = embedding_documents(query, bm_search, ranker, top_k)
        results = new_search.execute()
    else:  # bm25
        results = bm_search.execute()

    def get_relevance(annotation):
        if not annotation or annotation.split('-')[0] != args.topic_id:
            return 0
        return int(annotation.split('-')[1])

    # topic-relevance
    # ex: 321-2
    relevance = [get_relevance(hit.annotation) for hit in results]
    for hit in [hit for hit in results if hit.annotation.split('-')[0] == args.topic_id]:
        print(hit.annotation, hit.title, sep='\t')
    print(ndcg(relevance, top_k))


if __name__ == '__main__':
    main()
