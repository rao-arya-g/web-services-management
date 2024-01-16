from sklearn.cluster import KMeans, DBSCAN
from src.measure_performance.util import measure_silhouette_score
from src.pre_process_services.data_processor import tf_idf_encoder, word_embeddings_encoder, lda_model_encoder
from db_helper import DatabaseWrapper


def cluster_using_k_means(train_data_df, number_of_clusters):

    cluster_model = KMeans(n_clusters=number_of_clusters)
    s = cluster_model.fit(train_data_df)
    measure_silhouette_score(train_data_df, s.labels_, "K Means Accuracy")


def cluster_using_db_scan(train_data_df, _):
    cluster_model = DBSCAN(eps=0.3, min_samples=10)
    s = cluster_model.fit(train_data_df)
    measure_silhouette_score(train_data_df, s.labels_, "DBScan Accuracy")


def cluster_web_api_using_tf_idf(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()
    number_of_clusters = len(raw_data_df["category"].unique().tolist())
    feature_df = raw_data_df.drop('category', axis=1)

    x_train_encoded, x_test_encoded = tf_idf_encoder(feature_df, feature_df)
    cluster_using_k_means(x_train_encoded, number_of_clusters)
    cluster_using_db_scan(x_train_encoded, number_of_clusters)


def cluster_web_api_using_lda_model(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()
    feature_df = raw_data_df.drop('category', axis=1)

    number_of_clusters = len(raw_data_df["category"].unique().tolist())
    x_train_encoded, x_test_encoded = lda_model_encoder(feature_df, feature_df)
    cluster_using_k_means(x_train_encoded, number_of_clusters)
    cluster_using_db_scan(x_train_encoded, number_of_clusters)


def cluster_web_api_using_word_embeddings(database_wrapper=None):
    database_wrapper = database_wrapper or DatabaseWrapper()
    raw_data_df = database_wrapper.get_web_apis()
    raw_data_df = raw_data_df.dropna()
    feature_df = raw_data_df.drop('category', axis=1)

    x_train_encoded, x_test_encoded = word_embeddings_encoder(feature_df, feature_df)

    number_of_clusters = len(raw_data_df["category"].unique().tolist())
    cluster_using_k_means(x_train_encoded, number_of_clusters)
    cluster_using_db_scan(x_train_encoded, number_of_clusters)


if __name__ == '__main__':
    dw = DatabaseWrapper()
    print("Clustering Web Services using TF-IDF")
    cluster_web_api_using_tf_idf(dw)
    print("Clustering Web Services using Topic-Modeling")
    cluster_web_api_using_lda_model(dw)
    print("Clustering Web Services using Word Embedding")
    cluster_web_api_using_word_embeddings(dw)
