# Final Project: Minimum Wage
### Curtis Wilcox, Novia Wu, and Rachel Peng

### Thursday, May 13, 2021

#### COSI 132a (Spring 2021)

---

#### Description


#### Dependencies


#### Build Instructions


#### Run Instructions


#### Testing


***

#### System Design Description


***

PUT COMMENTS AND STUFF HERE (ONCE THEY'VE BEEN ALL WRITTEN BELOW)


***


#### Time Spent


#### Difficulty


#### General Comments

# CURTIS

This is a sample table:

Paragraph > Table > Insert table (command-option-T)

|      |      |      |      |
| ---- | ---- | ---- | ---- |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |




# NOVIA

- Tasks: 
    - Removal of boilerplate texts in queries 
    - Perform set difference of relevant query search and irrelevant query search to eliminate irrelevant documents
    - Ran most of the scores, before improvement and after each feature improvement
    - Computing tf-idf scores for each token to try to filter out less important words, so only important words are embedded
- Baseline scores:

|    NDCG  |   Title   |   Description   |   Narration   |
| ---- | ---- | ---- | ---- |
|   BM25 + def   |   0.767   |   0.836   |   0.708   |
|   sBERT + def   |   0.783   |   0.832   |   0.858   |
|   fastText + def   |   0.579   |   0.628   |  0.676    |



|    Ave Precision  |   Title   |   Description   |   Narration   |
| ---- | ---- | ---- | ---- |
|   BM25 + def   |   0.476   |   0.803   |   0.570   |
|   sBERT + def   |   0.709   |   0.717   |   0.832   |
|   fastText + def   |   0.331   |   0.502   |  0.457    |

- Final score after custom analyzer and custom query:

|    NDCG  |   Title   |   Description   |   Narration   |
| ---- | ---- | ---- | ---- |
|   BM25 + def   |   0.836   |   0.879   |  0.757    |
|   BM25 + n_gram   |   0.385   |   0.731   |   0.634   |
|   BM25 + whitespace   |   0.596   |   0.768   |   0.680   |
|   sBERT + def   |   0.857   |   0.896   |   0.822   |
|   sBERT + n_gram   |   0.860   |    0.843  |   0.700   |
|   sBERT + whitespace   |   0.842   |  0.839    |   0.716   |
|   fastText + def   |   0.606   |  0.783    |   0.817   |
|   fastText + n_gram   |   0.332   |   0.870   |   0.895   |
|   fastText + whitespace   |   0.430   |   0.958   |   0.865   |


|    Ave Precision  |   Title   |   Description   |   Narration   |
| ---- | ---- | ---- | ---- |
 BM25 + def   |   0.527   |    0.921  |   0.613   |
|   BM25 + n_gram   |   0.238   |  0.638    |   0.526   |
|   BM25 + whitespace   |   0.433   |   0.731   |  0.649    |
|   sBERT + def   |  0.787    |   0.944  |   0.715   |
|   sBERT + n_gram   |  0.978    |  0.864    |   0.656   |
|   sBERT + whitespace   |   0.877   |   0.832   |   0.742   |
|   fastText + def   |   0.344   |   0.724   |  0.698    |
|   fastText + n_gram   |   0.153   |    0.640  |   0.685   |
|   fastText + whitespace   |  0.266    |   0.840   |   0.724   |

- Time spent: around 10 hours researching, asking questions, implementing, scoring the retrieval methods
- Difficulty: 
    - Researching what is out there to improve the retrieval method took a lot of time, we were very confused in the beginning becase 
    of lack of exprience in the field 
    - Trying to figure out tf-idf for the embedding is very difficult. I was trying to improve by getting the tf-idf score
    for all the tokens, and then set a threshold, and "eliminate" the words that have little significance to the documents, 
    in hope to increase the accuracy by leaving more important words. However it did not work out. I understood theoretically
    how to compute and tf-idf, however, since the implementation is mainly in the embedding_service that is very complicated, 
    even after asking Jingxuan for help multiple times, I could not fix all the bugs that the system is telling me. I spent 
    a lot of time working on this part but unfortunately it did not work out.
    
# RACHEL







