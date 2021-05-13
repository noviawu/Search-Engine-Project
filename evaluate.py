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
    """
    create new "type" for Python / the argparser used for this CLI -- `unsigned int`
    :author: Curtis Wilcox
    :param x: value to convert to `unsigned int`, if is an `int` >= 0
    :return: integer version of `x`, iff `x` is an `int` >= 0
    :raise: argparse.ArgumentTypeError if `x` is not an `int` or `x` is an `int` < 0
    """
    if not isinstance(x, int) or int(x) < 0:
        raise argparse.ArgumentTypeError('Must be a number greater than or equal to zero.')
    return int(x)


def main():
    # create default ES connection
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    # these are the indices of each "column" once the XML file is parsed
    title = 0
    description = 1
    narration = 2

    parser = form_parser()
    args = parser.parse_args()

    topics = parse_wapo_topics(f'{Path("fp_data").joinpath("topics2018.xml")}')

    idx = title if args.query_type == 'title' else narration if args.query_type == 'narration' else description
    query = topics[args.topic_id][idx]

    # set analyzer
    if args.analyzer == 'n_gram':
        analyzer = 'n_gram'
    elif args.analyzer == 'whitespace':
        analyzer = 'whitespace'
    else:
        analyzer = 'default'

    # assured to be an int because types are enforced in the argparser
    top_k = int(args.top_k)

    bm_search = bm25_documents(query, analyzer, top_k)
    if args.vector_name:
        ranker = 'sbert' if args.vector_name == 'sbert_vector' else 'fasttext'
        new_search = embedding_documents(query, bm_search, ranker, top_k)
        results = new_search.execute()
    else:  # bm25
        results = bm_search.execute()

    def get_relevance(annotation):
        """
        inner function to grab just the number of relevance from the annotation
        :param annotation: annotation from document (in the form of `topic_id-relevance` [int-int])
        :return: relevance if relevant to current topic, otherwise 0
        """
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
