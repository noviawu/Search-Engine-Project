from collections import defaultdict
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Ids, Match
from embedding_service.client import EmbeddingClient
from example_query import generate_script_score_query
from flask import Flask, render_template, request

app = Flask(__name__)

RESULTS_PER_PAGE = 8

query = None
method = None
all_docs = []
documents = []
num_results = 0
results_back = 0


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

    query = request.form['query']
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


def get_documents(query, analyzer, ranker, results_back):
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    search = bm25_documents(query, analyzer, results_back)  # out here because need it for both
    if ranker == 'bm25':  # do no more work, just process the data
        return form_result_list(search.execute())
    else:  # rerank
        new_search = embedding_documents(query, search, ranker, results_back)
        return form_result_list(new_search.execute())


def bm25_documents(query, analyzer, results_back):
    if analyzer == 'default':
        q = Match(content={'query': query})
    elif analyzer == 'n_gram':
        q = Match(n_gram_custom_content={'query': query})
    elif analyzer == 'whitespace':
        q = Match(whitespace_custom_content={'query': query})
    return Search(using='default', index='wapo_docs_50k').query(q)[:results_back]


def embedding_documents(query, bm_search, ranker, results_back):
    ids = [hit.meta.id for hit in bm_search.execute()]
    q_match_ids = Ids(values=ids)

    encoder = EmbeddingClient(host='localhost', embedding_type=ranker)
    # noinspection PyTypeChecker
    embedding = encoder.encode([query], pooling="mean").tolist()[0]
    vector_name = ('ft' if ranker == 'fasttext' else 'sbert') + '_vector'
    q_vector = generate_script_score_query(embedding, vector_name)

    compound = (q_match_ids & q_vector)

    return Search(using='default', index='wapo_docs_50k').query(compound)[:results_back]
    # docs = s.execute()
    # return form_result_list(docs)


def form_result_list(docs):
    # TODO: put the 2s at top, then 1s, then 0s.
    paged_docs = defaultdict(list)
    i = 1
    for doc in docs:
        if len(paged_docs[i]) == RESULTS_PER_PAGE:
            i += 1
        paged_docs[i].append(
            {'doc_id': doc.doc_id, 'title': doc.title,
             'author': doc.author, 'date': doc.date, 'content': doc.content}
        )
    return {el['doc_id']: el for lst in paged_docs.values() for el in lst}, paged_docs


if __name__ == "__main__":
    app.run(debug=True, port=5000)
