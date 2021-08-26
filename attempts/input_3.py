import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

    # Using citation graph to correlate subjects
    cora_cites["class_cited"] = cora_cites["cited"].replace(cora_contet.index.values,
                                                            cora_contet.iloc[:, -1].values)
    cora_cites["class_citing"] = cora_cites["citing"].replace(cora_contet.index.values,
                                                              cora_contet.iloc[:, -1].values)

    # Counting citing/cited class (i.e. mapping the node degree by class) in the citation graph.
    # E.g.: paper2->paper1 == Neural_Networks -> Rule_Learning
    citing_class = cora_cites.groupby(["citing", "class_cited"]).size().unstack(fill_value=0)
    cited_class = cora_cites.groupby(["cited", "class_citing"]).size().unstack(fill_value=0)

    concat_citations = pd.concat([citing_class, cited_class], axis=1).fillna(0)

    # sum the node output and node degree inputs degree
    concat_citations_scaled = concat_citations.groupby(lambda x: x, axis=1).sum()
    concat_citations_scaled[:] = MinMaxScaler().fit_transform(concat_citations_scaled.T).T

    # Created binary classes based on the node degree
    concat_citations_binary = concat_citations_scaled.astype('int32')
    # Encode the classes names into target labels (0,1,2...N)

    le = LabelEncoder()
    targets = le.fit_transform(cora_contet.iloc[:, -1])

    # Term-Freq dict
    tf = cora_contet.iloc[:, :-1]

    # Add concat_citations to the training_set
    training_set = pd.concat([tf, concat_citations_binary], axis=1)

    # The dataset has imbalanced data. So, the train and test split will be stratified
    # Using [Term-Freq dict;concat_citations]
    X_train, X_test, y_train, y_test = train_test_split(
        training_set, targets, test_size=0.2, random_state=RANDOM_SEED,
        stratify=targets
    )

    skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

    clfs = [
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("BernoulliNB", BernoulliNB(binarize=None)),

    ]

    for name, clf in clfs:
        cross_score = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"{name}. Dev_acc:{cross_score.mean()}")
