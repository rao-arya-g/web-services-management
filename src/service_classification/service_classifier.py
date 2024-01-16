from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from src.measure_performance.util import measure_accuracy
from src.pre_process_services.pre_processor import tf_idf_encoder, word_embeddings_encoder, lda_model_encoder
from db_helper import DatabaseWrapper


def decision_tree_classifier(x_train, x_test, y_train, y_test):
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)

    measure_accuracy(y_test, y_prediction, "Decision Tree Accuracy")


def nearest_neighbor_classifier(x_train, x_test, y_train, y_test):
    classifier = KNeighborsClassifier()
    classifier = classifier.fit(x_train, y_train)

    y_prediction = classifier.predict(x_test)

    measure_accuracy(y_test, y_prediction, "Nearest Neighbor Accuracy")


def naive_bayes_classifier(x_train, x_test, y_train, y_test):

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    measure_accuracy(y_test, y_prediction, "Naive Bayes Accuracy")


def classify_web_api_using_tf_idf(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()

    feature_df = raw_data_df.drop('category', axis=1)
    target_df = raw_data_df['category']

    x_train, x_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=1)

    x_train_encoded, x_test_encoded = tf_idf_encoder(x_train, x_test)
    decision_tree_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    nearest_neighbor_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    naive_bayes_classifier(x_train_encoded, x_test_encoded, y_train, y_test)


def classify_web_api_using_lda_model(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()
    feature_df = raw_data_df.drop('category', axis=1)
    target_df = raw_data_df['category']
    x_train, x_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=1)

    x_train_encoded, x_test_encoded = lda_model_encoder(x_train, x_test)
    decision_tree_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    nearest_neighbor_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    naive_bayes_classifier(x_train_encoded, x_test_encoded, y_train, y_test)


def classify_web_api_using_word_embeddings(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()
    feature_df = raw_data_df.drop('category', axis=1)
    target_df = raw_data_df['category']

    x_train, x_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=1)

    x_train_encoded, x_test_encoded = word_embeddings_encoder(x_train, x_test)

    decision_tree_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    nearest_neighbor_classifier(x_train_encoded, x_test_encoded, y_train, y_test)
    naive_bayes_classifier(x_train_encoded, x_test_encoded, y_train, y_test)


if __name__ == '__main__':
    dw = DatabaseWrapper()
    print("Classifying Web Services using TF-IDF")
    classify_web_api_using_tf_idf(dw)
    print("Classifying Web Services using Topic-Modeling")
    classify_web_api_using_lda_model(dw)
    print("Classifying Web Services using Word Embedding")
    classify_web_api_using_word_embeddings(dw)
