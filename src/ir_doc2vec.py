from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from util import remove_url
from util import remove_punct
from util import remove_tag
from util import output_ranks
import spacy
import gensim.downloader


nlp = spacy.load("en_core_web_sm")
glove_vectors = gensim.downloader.load('glove-twitter-25')


def get_ranks(document_set, query_set, model):
    ranks = {}

    for query_index, qry in enumerate(query_set['text']):
        for document_index, doc in enumerate(document_set['text']):
            query_id = query_set['id'][query_index]
            document_id = document_set['id'][document_index]

            if query_id in ranks:
                ranks[query_id][document_id] = model.similarity_unseen_docs(
                    qry, doc)
            else:
                ranks[query_id] = {
                    document_id: model.similarity_unseen_docs(qry, doc)}

    for query, relevant_documents in ranks.items():
        ranks[query] = dict(
            sorted(relevant_documents.items(), key=lambda item: item[1], reverse=True))

    return ranks


def get_document_set(documents):
    document_set = {
        'id': [],
        'text': [],
        'tagged_text': []
    }

    for i, document in enumerate(documents):
        document_set['id'].append(document['id'])
        document_set['text'].append(document['doc'])
        document_set['tagged_text'].append(
            TaggedDocument(document['doc'], [i]))

    return document_set


def get_query_set(queries):
    query_set = {
        'id': [],
        'text': []
    }

    for query in queries:
        query_set['id'].append(query['num'])
        query_set['text'].append(query['title'])

    return query_set


def process_documents(document_file_name):
    document_file = open(document_file_name, encoding="utf-8")
    document_str_list = remove_punct(
        remove_url(document_file.read())).split("\n")

    documents = []

    for document_str in document_str_list:
        if document_str.isascii():
            document = process_document(document_str)

            if document is not None:
                documents.append(document)

    return documents


def process_document(document_str):
    document_list = document_str.split('\t')

    id = document_list[0]
    text = process_text(document_list[1])

    if len(text) >= 5:
        document = {
            'id': id,
            'doc': text,
        }

        return document
    else:
        return None


def process_queries(query_file_name):
    query_file = open(query_file_name, encoding="utf-8")
    query_str_list = query_file.read().split("\n\n")

    queries = []

    for query_str in query_str_list:
        query = process_query(query_str)
        queries.append(query)

    return queries


def process_query(query_str):
    query_list = query_str.split('\n')

    num = int(remove_tag(query_list[1], 'num')[10:])
    time = datetime.strptime(remove_tag(
        query_list[3], 'querytime'), '%a %b %d %H:%M:%S %z %Y')
    tweet_time = remove_tag(query_list[4], 'querytweettime')

    terms = process_text(remove_tag(query_list[2], 'title'))

    text = []

    for term in terms:
        text.append(term)

        try:
            similar_word = glove_vectors.most_similar(term, topn=1)[0][0]
            text.append(similar_word)
        except:
            print("No similar word found for term " + term)

    query = {
        'num': num,
        'title': text,
        'querytime': time,
        'querytweettime': tweet_time,
    }

    return query


def process_text(text):
    tokens = nlp(text)
    terms = []

    for token in tokens:
        if not token.is_stop:
            term = token.lemma_
            term = term.lower().strip()

            if len(term) >= 2:
                terms.append(term)

    return terms


# To run this experiment, uncomment the following code after adding appropriate configerations.
# Add the paths to document file, query file, result file to document_file_name, query_file_name, result_file_name.
# Add the experiment name to run_id.

# document_file_name = ""
# query_file_name = ""
# result_file_name = ""
# run_id = ""

# documents = process_documents(document_file_name)
# queries = process_queries(query_file_name)

# document_set = get_document_set(documents)
# query_set = get_query_set(queries)


# model = Doc2Vec(document_set['tagged_text'],
#                 vector_size=5, window=2, min_count=1, workers=4)

# ranks = get_ranks(document_set, query_set, model)

# output_ranks(result_file_name, ranks, run_id)