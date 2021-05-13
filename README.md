# Final Project: Federal Minimum Wage Increase (TREC #816)
### Curtis Wilcox, Novia Wu, and Rachel Peng

### Thursday, May 13, 2021

#### COSI 132a (Spring 2021)

---

#### Preliminary

##### Team member submitting code



##### Title

Federal Minimum Wage Increase

##### Description

Find descriptions of the actions and reactions of either the President or Congress to increase the U.S. federal minimum wage.

##### Narrative

Relevant documents include descriptions of advocacy or actions (or lack thereof) taken by the President or Congress to increase the U.S. federal minimum wage, including increases for government contract workers. Analyses and discussions of pros and cons of an increase by talking heads is not relevant.

##### Queries

- **Updated Description:** actions and reactions of President or Congress to increase U.S. federal minimum wage
- **Updated Narrative:** advocacy or actions (or lack thereof) by the President or Congress to increase the U.S. federal minimum wage, government contract workers

##### (Brief) Summary

Our team did multiple things to optimize the user's retrieval of documents that are pertinent to this subject.

We created two new custom analyzers -- one that used the `trigram` tokenizer and one that used the `whitespace` tokenizer (both analyzers have filters based on lowercase letters, stopwords, asciifolding, and use the porter stemmer).

We implemented a synonym mechanism.

The Flask web app will automatically suggest queries for the user based on what they have typed in at any given moment. The suggestions are based off of the titles of the documents.

---

#### Description

##### Flask Application

This program allows a user to enter a search query and filter through several thousand documents by seeing which documents contains any of the terms searched for (with some exceptions). The user can see eight results per page (where each result is the title and a snippet of the document's content) and navigate through the pages, and also click on links to each document's full content.

##### Command Line Interface (CLI)

There is also a provided CLI. The user can enter queries to obtain the `Normalized Discount Cumulative Gain` (`NDCG`) score and the `Average Precision` of a query by specifying the following information:

- `--index_name`: the name of the index;
- `--topic_id`: the ID number of the topic;
- `--query_type`: the query (which will be the `title`, `description`, or `narration` of the topic specified);
- `--vector_name`: optionally, the vector to use for reranking (`sbert_vector` or `ft_vector`);
- `--top_k`: the number of documents to retrieve; and
- `--analyzer`: which analyzer to use (either `default`, `n_gram`, or `whitespace`).

#### Dependencies

- `Python 3` (specifically,  `Python 3.8`)
  - Installed when creating the virtual environment (venv). Using `miniconda`, the command was `conda create -n cosi132a python=3.8`
- `elasticsearch, elasticsearch-dsl, sentence-transformers, flask, numpy, zmq`
  - Installed using `pip` in the activated venv (`pip install -r requirements.txt`, where `requirements.txt` contains each of the previously mentioned requirements, separated by a newline, and nothing else)

#### Build Instructions

0. Note: Steps 1-8 should be run on the command line. Please also note that these steps are tried-and-true on MacOS. Nobody on this team uses Windows or Linux, so we apologize if they do not work on those machines.

1. ```shell
   conda create -n my_environment python=3.8  # my_environment could be anything you want
   ```

2. ```shell
   conda activate my_environment
   ```

3. ```shell
   pip install -r requirements.txt  # requirements.txt should be provided
   ```

4. ```shell
   python load_es_index.py --index_name wapo_docs_50k
   --wapo_path fp_data/subset_wapo_50k_sbert_ft_filtered.jl
   ```
   
5. ```shell
   python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3
   ```

6. ```shell
   python -m embedding_service.server --embedding fasttext  
   --model fp_data/wiki-news-300d-1M-subword.vec
   ```

7. ```shell
   python fp.py --run
   ```

8. Navigate to `localhost:5000` in your browser of choice.

#### Run Instructions

##### Flask Application

1. Enter any string into the search bar provided at the top of the screen and select which search method you would like to utilize, followed by the number of results you want returned. Press the `Search` button or hit the `return` button on your keyboard when you are satisfied with your query.
2. A maximum of eight results will be displayed on a page. If your search query had more than eight hits, there will be a `Next Page` button at the bottom that can be pressed to navigate to the next page.
   1. If you enter nothing into your query, you will receive no results.
   2. There is no button on the website to go backwards: you must use the browser's `back` button to move in that direction.
   3. The documents will be displayed in order of supposed relevancy.
3. Each result shows the title of the document that matched and the first 150 characters of the document. To see more information or content of a specific document, click on the title (as it is a link) to take you to the appropriate page.
4. If you wish to enter a new query, you need only enter a new query in the search bar at the top of the results page. There is no search bar when viewing information specific to one document. For convenience, the "active" query will be presented in the search bar, but you need only change the text and run a new query for new results.

##### CLI

1. Start the query with `python evaluate.py` (the venv from above should of course be activated).
2. After specifying the filename, add the arguments that are explained in the `Description / CLI` section above.
3. View the data at your leisure! 

***

#### System Design Description

The system that we created for this final project runs primarily off of two files. The `fp.py` file runs the Flask app, and it contains logic for obtaining information that the system will utilize (such as documents to filter through, select, and read) and passing it to the relevant recipients (that is, the HTML pages seen by the user). When the documents are retrieved, they are held in a dictionary that maps a list of documents to the page number that they will appear on. The `evaluate.py` file runs the CLI.

`utils.py` contains the logic for reading in the information in the documents (and extracting the information requested by the assignment). That would be reading in the `wapo` data from the `.db` file and also the wapo topics from the `.xml` file. There is also a helper function for taking in an iterable and returning the first unique `n` elements that have a length greater than `min_length`.

`evaluate.py` contains the logic for a command-line interface that will display the `NDCG@{top_k}` score for a query. The data can be queried by using the title, description, or narration of a specified wapo topic. The script must also be supplied the number of documents to return and whether the default analyzer or the custom analyzer will be used, as well as whether or not the results should be reranked (using the `sBERT` embeddings or the `fastText` embeddings if so). More information is provided in `Description/CLI` and `Run Instructions/CLI`.

`example_analyzer.py`, `example_embedding.py`, and `example_query.py` each provide examples of the analyzers/embeddings/queries that can be used.

`load_es_index.py` reads in the wapo data and constructs the index for accessing its information.

`metrics.py` contains logic for calculuating the `NDCG` score and the `Average Precision` of a query.

`home.html` is the "index" file for the site. It presents a search bar to the user and allows them to enter a query to find relevant documents, by way of selecting the search method (reranking, which analyzer to use, number of results to retrieve).

`results.html` displays a list of documents that match the query. A maximum of eight results can be displayed at any one time, so if there are more than eight results there will be a "next page" button for the user to click to navigate through the list. The same search bar is displayed to find other documents. There is no "previous page" button for the user (instead, they may use their browser's "back" button). Each result is comprised of two parts: the first is the title of the document in a link format, so that the user may click on it to access the document's full information; the second is the first 150 characters of the document's content, followed by an ellipse (…). Each result is numbered `1` through however many results are returned.

`doc.html` displays specific information for a selected document (the title, the author, the date the document was published, and the content). The only way to navigate back is by using the browser's back button.


***

PUT COMMENTS AND STUFF HERE (ONCE THEY'VE BEEN ALL WRITTEN BELOW)

#### Breakdown by Teammate

##### Curtis

I predominantly worked on the autosuggestion feature demonstrated in the Flask application. This was more difficult than I had originally anticipated because of the shortage of clear documentation. There were a healthy amount of examples on the `ElasticSearch` site, but they were all `cURL` examples and not `ElasticSearch DSL` (`DSL`) examples, and I did not find it terribly easy to parse from `cURL` to `DSL` (that is to say, nigh-impossible). I eventually determined that a new field needed to be added to the `BaseDoc` class (of type `elasticsearch_dsl.Completion`), and that a field needed to be added when constructing the index. 

The field added to the index needed to be a list of strings (where each string was a suggestion for the document), which I decided to create by removing non-alphabetic (or space) characters from the title, then taking the first six words that are unique and not one character long, and generating all permutations of the words in that string. Unfortunately, the index now takes about 20 minutes to build.

In the HTML files that include a search bar (`home.html` and `results.html`), there is an additional `JavaScript` / `AJAX` script included in the `head`, as well as two `jquery` scripts. The script adds an autocomplete aspect to the search bar where the user enters, and handles sending the data to the `search` function, as well as getting the data back from the search function and sending it to the provided `autocomplete` functionality. This website-specific logic was taken from a website cited in the `.html` files and slightly adapted to fit the needs of this assignment.

The Flask-side display of success comes in the form of the `search` function in `fp.py`. The search term is obtained from the `HTML` form and a connection is made to ElasticSearch. Then use the `Search.suggest` method, pass it a title to return the suggestions under, the term to search over, and a `completion` argument that specifies that the field with information for completing the autocompletion in the form of a dictionary (e.g., `{'field': 'field_from_doc_template'}`). The responses are then "jsonified" (`jsonify` is a Flask method that wraps over Python's `json.dumps` to turn the result into a `flask.Response`) and sent back to the web-server side of things.

Ideally, the suggestions would not be based solely on the first six unique two-or-more-character words in the title of the document, and have more to do with parts of the content of the document being suggested. However, since generating permutations is expensive, and it took so long to paw through layers of difficult documentation, StackOverflow questions, and GitHub Issues, and get all of the parts to work together, this is the method that was selected.

##### Novia



##### Rachel




***


#### Time Spent


#### Difficulty


#### General Comments

# CURTIS

- `utils.py` -- `first_unique_n`

- `fp.py` -- `search`
  
- `index.py` -- `_populate_doc`

- add `Completion` to `doc_template.py`

This is a sample table:

Paragraph > Table > Insert table (command-option-T)

|      |      |      |      |
| ---- | ---- | ---- | ---- |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |





# NOVIA













# RACHEL







