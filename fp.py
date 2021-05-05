from collections import defaultdict
from typing import List

from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Ids, Match, MatchAll
from embedding_service.client import EmbeddingClient
from example_query import generate_script_score_query
from flask import Flask, render_template, request, jsonify
import json, os
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import wordnet
from embedding_service.text_processing import TextProcessing


app = Flask(__name__)

RESULTS_PER_PAGE = 8

query = None
method = None
all_docs = []
documents = []
num_results = 0
results_back = 0

# text processor for query using customized processor
text_processor = TextProcessing.from_nltk()


# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    """
    result page
    :return: page that shows the first eight results (or all results if there were <= eight)
    """
    global query
    global method
    global all_docs
    global documents
    global num_results
    global results_back

    query_input = request.form['query']
    # query processing
    query_list: List[str] = query_input.split(" ")
    query_n = normalize_query(query_list)
    query = get_synonyms(query_n)
    method = request.form['method']
    ranker = method.split('-')[0]
    analyzer = method.split('-')[1]
    results_back = int(request.form['num-results'])
    all_docs, documents = get_documents(query, analyzer, ranker, results_back)

    # calculate the total number of results found
    num_results = sum(len(ds) for ds in documents.values())

    return render_template("results.html", query=query, docs=documents[1], num_results=num_results,
                           page_id=1, method=method, res_per_page=RESULTS_PER_PAGE,
                           results_back=results_back,
                           last_page=(num_results == len(documents[1])))  # add variables as you wish


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    """
    "next page" to show more results
    :param page_id: the page to display
    :return: next page of results
    """
    global query  # ugh
    global documents  # blech
    global num_results  # gross
    global method
    global results_back
    # print(documents[page_id])
    return render_template("results.html", query=query, docs=documents[page_id],
                           num_results=num_results, page_id=page_id, method=method,
                           res_per_page=RESULTS_PER_PAGE, results_back=results_back,
                           last_page=(num_results <= page_id * RESULTS_PER_PAGE))  # add variables as you wish


# document page
@app.route("/doc_data/<doc_id>")
def doc_data(doc_id):
    """
    document page
    :param doc_id: document to display
    :return: page containing information on the selected document
    """
    global all_docs
    return render_template("doc.html", info=all_docs[doc_id])


# search "page" (autocompletion)
@app.route('/search', methods=['POST'])
def search():
    """
    search "page" to give the auto-completion when typing in suggestions
    :return: jsonified list of possible autocompletion matches
    """
    term = request.form['q']
    print(term)
    connections.create_connection(hosts=['localhost'], timeout=100, alias='default')
    q = Match(title={'query': term})
    s = Search(using='default', index='wapo_docs_50k').query(q)
    suggest = s.suggest('autocomplete_suggestion', term, term={'field': 'content'})
    response = suggest.execute()
    resp = jsonify([hit.title for hit in response])
    resp.status_code = 200
    return resp


def get_documents(query, analyzer, ranker, results_back):
    """
    get the documents that match the query, using the specified analyzer and reranker (if applicable)
    :param query: query to match on
    :param analyzer: analyzer to use
    :param ranker: reranker to use
    :param results_back: how many results to return
    :return: list of documents that match parameters
    """
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    search = bm25_documents(query, analyzer, results_back)  # out here because need it for both
    if ranker == 'bm25':  # do no more work, just process the fp_data
        return form_result_list(search.execute())
    else:  # rerank
        new_search = embedding_documents(query, search, ranker, results_back)
        return form_result_list(new_search.execute())


def bm25_documents(query, analyzer, results_back):
    """
    use the BM25 algorithm to find documents that match the query
    :param query: query to match on
    :param analyzer: analyzer to use
    :param results_back: how many results to return
    :return: `Search` object to match documents
    """
    if analyzer == 'default':
        q = Match(content={'query': query})
    elif analyzer == 'n_gram':
        q = Match(n_gram_custom_content={'query': query})
    elif analyzer == 'whitespace':
        q = Match(whitespace_custom_content={'query': query})
    return Search(using='default', index='wapo_docs_50k').query(q)[:results_back]


def embedding_documents(query, bm_search, ranker, results_back):
    """
    use the selected reranker to rerank documents found by BM25
    :param query: query to match on
    :param bm_search: results of the BM25 algorithm
    :param ranker: reranker to use
    :param results_back: number of results to return
    :return: `Search` object to match on reranked documents
    """
    ids = [hit.meta.id for hit in bm_search.execute()]
    q_match_ids = Ids(values=ids)

    encoder = EmbeddingClient(host='localhost', embedding_type=ranker)
    # noinspection PyTypeChecker
    embedding = encoder.encode([query], pooling="mean").tolist()[0]
    vector_name = ('ft' if ranker == 'fasttext' else 'sbert') + '_vector'
    q_vector = generate_script_score_query(embedding, vector_name)

    compound = (q_match_ids & q_vector)

    return Search(using='default', index='wapo_docs_50k').query(compound)[:results_back]


def form_result_list(docs):
    """
    put all documents matched in a dictionary to easily render documents per page for the user
    :param docs: documents that were matched / to display
    :return: tuple: dictionary of doc id to relevant document, defaultdict of docs by page they appear on,
                    and sorted by relevance (most relevant appear first)
    """
    paged_docs = defaultdict(list)
    i = 1
    # print([(hit.title, hit.annotation) for hit in sorted(docs, key=get_hit_key, reverse=True)])
    # docs = sorted(docs, key=get_hit_key, reverse=True)  # maybe lose this
    for doc in docs:
        if len(paged_docs[i]) == RESULTS_PER_PAGE:
            i += 1
        paged_docs[i].append(
            {'doc_id': doc.doc_id, 'title': doc.title,
             'author': doc.author, 'date': doc.date, 'content': doc.content,
             'annotation': doc.annotation}
        )

    return {el['doc_id']: el for lst in paged_docs.values() for el in lst}, paged_docs


def get_synonyms(query_list: List[str]) -> str:
    """
    get a list of synonyms for the user input query and return
    """
    synonyms = set()
    synonyms.add(query)
    for q in query_list:
        syn = wordnet.synsets(q)
        for s in syn:
            for lm in s.lemmas():
                synonyms.append(lm.name())  # adding into synonyms
    list_of_strings = [str(s) for s in synonyms]
    joined_string = " ".join(list_of_strings)
    return joined_string


def normalize_query(query_list: List[str]) -> str:
    """
    return a normalized query with the customized text processor
    """
    normalized_q = []
    for q in query_list:
        normalized_q.append(text_processor.normalize(q))
    return "".join(normalized_q)


# def get_hit_key(hit):
#     """
#     helper function for `form_result_list` -- basically, get the relevance of the document
#     :param hit: hit to get relevance from
#     :return: how relevant the document is (0, 1, 2)
#     """
#     # if hit.annotation:
#     #     return int(hit.annotation.split('-')[1])
#     # return 0
#     return int(hit.annotation.split('-')[1]) if hit.annotation else 0


if __name__ == "__main__":
    app.run(debug=True, port=5000)
