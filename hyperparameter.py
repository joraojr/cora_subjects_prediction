import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

# To ensure reproducibility
RANDOM_SEED = 42

# The use of Term-Feq alone (baseline.py) was the best input. I selected it to be used in the hyperparameter tuning
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

    clfs = [
        ("Decision Tree", GridSearchCV(
            DecisionTreeClassifier(), param_grid=
            {
                'max_features': ['auto', 'sqrt'],
                "criterion": ["gini", "entropy"],
                'max_depth': [10, 20, 50, 90],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            scoring="accuracy",
            n_jobs=-1,
            cv=skf
        ),
         ),
        ("Random Forest", GridSearchCV(
            RandomForestClassifier(),
            {
                "criterion": ["gini", "entropy"],
                'n_estimators': [2, 5, 10, 20],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, 50, 90],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            scoring="accuracy",
            n_jobs=-1,
            cv=skf
        ),
         ),
        ("BernoulliNB", GridSearchCV(
            BernoulliNB(),
            {
                'alpha': np.linspace(0.01, 1, num=100),
                'fit_prior': [True, False],
                'binarize': [None]
            }, scoring="accuracy",
            n_jobs=-1,
            cv=skf
        )
         ),

    ]

    print("Starting GridSearch:")
    for name, clf in clfs:
        clf.fit(X_train, y_train)
        print(f"-{name}:  Grid best_score: {clf.best_score_}. Grid best_estimator: {clf.best_estimator_}")
