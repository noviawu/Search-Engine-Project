from elasticsearch_dsl import (  # type: ignore
    Document,
    Text,
    Keyword,
    DenseVector,
    Date,
    token_filter,
    analyzer,
    tokenizer,
    Completion
)

n_gram = analyzer(
    'n_gram',
    tokenizer=tokenizer('trigram', 'ngram', min_gram=3, max_gram=4),
    filter=['lowercase', 'stop', 'asciifolding', 'porter_stem'],
)

whitespace = analyzer(
    'whitespace',
    tokenizer='whitespace',
    filter=['lowercase', 'stop', 'asciifolding', 'snowball'],
)


class BaseDoc(Document):
    """
    wapo document mapping structure
    """
    # we want to treat the doc_id as a Keyword (its value won't be tokenized or normalized).
    doc_id = Keyword()

    title = Text()  # by default, Text field will be applied a standard analyzer at both index and search time

    author = Text()

    content = Text(analyzer="standard")  # we can also set the standard analyzer explicitly

    n_gram_custom_content = Text(analyzer=n_gram)

    whitespace_custom_content = Text(analyzer=whitespace)

    date = Date(format="yyyy/MM/dd")  # Date field can be searched by special queries such as a range query.

    annotation = Text()

    ft_vector = DenseVector(dims=300)  # fasttext embedding in the DenseVector field

    sbert_vector = DenseVector(dims=768)  # sentence BERT embedding in the DenseVector field

    title_suggest = Completion()  # for use in autocompletion when searching

    def save(self, *args, **kwargs):
        """
        save an instance of this document mapping in the index
        this function is not called because we are doing bulk insertion to the index in the index.py
        """
        return super(BaseDoc, self).save(*args, **kwargs)


if __name__ == "__main__":
    pass
