from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
import gensim.models as models
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation


class LemmaTokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t.isalpha()]


def merge_tags(row):
    if isinstance(row["Tags"], str):
        return row["Tags"]
    else:
        return " ".join(row["Tags"])


def get_all_descriptions(raw_data_df):
    processed_df = raw_data_df.copy()
    processed_df["combined_tags"] = processed_df.apply(merge_tags, axis=1)
    processed_df["combined_column"] = processed_df[["combined_tags", "label", "summary"]].agg(' '.join, axis=1)
    all_descriptions = processed_df["combined_column"].values
    return all_descriptions


def tf_idf_encoder(train_data_df, test_data_df):
    stop_words_list = stopwords.words('english')
    tokenizer = LemmaTokenizer()
    stop_words_list = tokenizer(' '.join(stop_words_list))

    train_vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=250, stop_words=stop_words_list)
    train_descriptions = get_all_descriptions(train_data_df)
    tf = train_vectorizer.fit_transform(train_descriptions)
    train_encoded_df = pd.DataFrame(tf.toarray())

    test_vectorizer = TfidfVectorizer(vocabulary=train_vectorizer.vocabulary_, tokenizer=LemmaTokenizer(),
                                      stop_words=stop_words_list)
    test_descriptions = get_all_descriptions(test_data_df)
    tf = test_vectorizer.fit_transform(test_descriptions)
    test_encoded_df = pd.DataFrame(tf.toarray())

    return train_encoded_df, test_encoded_df


def word_embeddings_encoder(train_data_df, test_data_df):

    cores = multiprocessing.cpu_count()
    train_descriptions = get_all_descriptions(train_data_df)
    all_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_descriptions)]
    model = models.Doc2Vec(vector_size=10, window=2, workers=cores)
    model.build_vocab(all_documents)
    model.train(all_documents, total_examples=model.corpus_count, epochs=10)
    test_descriptions = get_all_descriptions(test_data_df)
    train_vectors = []
    test_vectors = []
    for entry in test_descriptions:
        test_vectors.append(model.infer_vector(entry.split()))

    for entry in train_descriptions:
        train_vectors.append(model.infer_vector(entry.split()))

    return train_vectors, test_vectors


def lda_model_encoder(train_data_df, test_data_df):
    stop_words_list = stopwords.words('english')
    tokenizer = LemmaTokenizer()
    stop_words_list = tokenizer(' '.join(stop_words_list))

    number_of_topics = 200
    lda_model = LatentDirichletAllocation(n_components=number_of_topics,
                                          max_iter=100)

    train_vectorizer = CountVectorizer(tokenizer=tokenizer, max_features=750, stop_words=stop_words_list)
    train_descriptions = get_all_descriptions(train_data_df)
    tf_train = train_vectorizer.fit_transform(train_descriptions)

    test_vectorizer = CountVectorizer(vocabulary=train_vectorizer.vocabulary_, tokenizer=LemmaTokenizer(),
                                      stop_words=stop_words_list)
    test_descriptions = get_all_descriptions(test_data_df)
    tf_test = test_vectorizer.fit_transform(test_descriptions)

    train_encoded_df = lda_model.fit_transform(tf_train)
    test_encoded_df = lda_model.transform(tf_test)

    return train_encoded_df, test_encoded_df
