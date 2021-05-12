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
    if not isinstance(0, int) or int(x) < 0:
        raise argparse.ArgumentTypeError('Must be a number greater than or equal to zero.')
    return int(x)


def main():
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")

    title = 0
    description = 1
    narration = 2

    parser = form_parser()
    args = parser.parse_args()

    topics = parse_wapo_topics(f'{Path("fp_data").joinpath("topics2018.xml")}')

    idx = title if args.query_type == 'title' else narration if args.query_type == 'narration' else description
    query = topics[args.topic_id][idx]

    if args.analyzer == 'n_gram':
        analyzer = 'n_gram'
    elif args.analyzer == 'whitespace':
        analyzer = 'whitespace'
    else:
        analyzer = 'default'

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
    print("NDCG: ", ndcg(relevance, top_k))
    print("ave precision: ", average_precision(relevance))


if __name__ == '__main__':
    main()
