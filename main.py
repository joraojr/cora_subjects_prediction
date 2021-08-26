import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import plot_confusion_matrix, accuracy_score


# To ensure reproducibility
RANDOM_SEED = 42

if __name__ == "__main__":
    cora_cites = pd.read_csv("cora/cora.cites", sep="\t", header=None, names=["cited", "citing"])
    cora_contet = pd.read_csv("cora/cora.content", sep="\t", header=None, index_col=0)

    print("=" * 10 + "Cora_cites head" + "=" * 10)
    print(cora_cites.head())
    print("=" * 40)
    print("=" * 10 + "Cora_content head" + "=" * 10)
    print(cora_contet.head())
    print("=" * 40)

    print("=" * 10 + "Subject distribution" + "=" * 10)
    print(cora_contet.iloc[:, -1].value_counts(normalize=True))
    print("=" * 40)

    # Encode the classes names into target labels (0,1,2...N)
    le = LabelEncoder()
    targets = le.fit_transform(cora_contet.iloc[:, -1])

    # Term-Freq dict
    tf = cora_contet.iloc[:, :-1]

    # Analyzing Term-Freq correlation to remove strong correlated features
    tf_corr = tf.corr().abs()
    upper_tri = tf_corr.where(np.triu(np.ones(tf_corr.shape), k=1).astype(bool))

    # Correlation >=0.8 can 'confuse the model' and were extracted
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.8)]
    print(f"Strong correlated columns: {to_drop}")

    tf_norm = tf.drop(tf.columns[to_drop], axis=1)
    print(tf_norm.head())

    # The dataset has imbalanced data. So, the train and test split will be stratified
    # Only using Term-Freq dict
    X_train, X_test, y_train, y_test = train_test_split(
        tf, targets, test_size=0.2, random_state=RANDOM_SEED,
        stratify=targets
    )

    skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

    # The dataset is composed of a Term-Found (0/1)dict.
    # BernoulliNB usually works good over binary/boolean discrete classification.
    # It will be my baseline:

    # Best model after hyperparameter tunning
    clf = BernoulliNB(alpha=0.13, binarize=None)

    cross_score = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Cross_validation mean accuracy :{cross_score.mean()}")

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(f"Test accuracy :{accuracy_score(y_test, predictions)}")

    with open("./predictions.tsv", "w") as output:
        for id, pred in zip(X_test.index.values, le.inverse_transform(predictions)):
            output.write(f"{id} \t {pred} \n")

    fig, ax = plt.subplots(figsize=(15, 15))
    plot_confusion_matrix(clf, X_test, y_test, display_labels=le.classes_, xticks_rotation="vertical", normalize="true",
                          ax=ax)
    plt.savefig("./matrix.png", dpi=300)
