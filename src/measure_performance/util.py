from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def measure_accuracy(y_test, y_prediction, message):
    score = accuracy_score(y_test, y_prediction)
    print("\t" + message + " is " + str(round(score, 4) * 100) + "%")


def measure_ami(actual_label, ground_truth, message):
    label_encoder = LabelEncoder()
    ground_truth_label = label_encoder.fit_transform(ground_truth)
    score = adjusted_mutual_info_score(ground_truth_label, actual_label)
    print("\t" + message + " is " + str(round(score, 4) * 100) + "%")


def measure_silhouette_score(train_data_df, labels, message):
    try:
        score = silhouette_score(train_data_df, labels)
        print("\t" + message + " is " + str(round(score, 4) * 100) + "%")
    except ValueError as _:
        print(message + " Cannot be calculated due to less of number of labels than expected")
