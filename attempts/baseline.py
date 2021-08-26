import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

# To ensure reproducibility
RANDOM_SEED = 42

if __name__ == "__main__":
    cora_cites = pd.read_csv("../cora/cora.cites", sep="\t", header=None, names=["cited", "citing"])
    cora_contet = pd.read_csv("../cora/cora.content", sep="\t", header=None, index_col=0)

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

    clfs = [
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("BernoulliNB", BernoulliNB()),

    ]

    for name, clf in clfs:
        cross_score = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"{name}. Dev ACC:{cross_score.mean()}")
