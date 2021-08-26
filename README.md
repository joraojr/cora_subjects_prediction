# Subjects prediction on Cora dataset

This project explores Cora dataset features to build a machine learning model to classify papers subjects.

## Dataset

The Cora dataset consists of 2708 scientific publications classified into one of seven classes (`Case_Based`
, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`)
. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector
indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique
words. The README file in the dataset provides more details.

## Files:

- ``main.py``: The main script containing the best model after hyperparameter tuning.
- ``hyperparameter.py``: A script containing the hyperparameter step.
- ``attempts/``: This directory has differents attempts of preprocessing and input generation:
    - ``baseline.py``: The model baseline using only the Terms/Appearance in a paper

    - ``input_1.py``: Inclusion of citation graph data as input. The input and output degrees of each graph node were used
      as additional inputs (cited / citing)
    - ``input_2.py``: The classes of the papers that were cited/citing were used as additional columns. E.g.: paper2->
      paper1 == Neural_Networks -> Rule_Learning. The importance of each paper was normalized.
    - ``input_3.py``: The classes of the papers that were cited/citing were used as additional columns however, binarized
      between 0 and 1.
    - ``input_4.py``: Attempt to use Graph coloring to detect communities on the citation graph.

- ``predictions.tsv``: Predictions file

- ``matrix.png``: Confusion matrix presenting the results for each subject

## How to run?

Install the pip requirements:

``pip install -r requirements.txt``

The final model with predictions is on the main.py:

``python main.py``

## Observations

- The model that achieved better results was the BernoulliNB using only the Term/Presence. BernoulliNB usually works
  good over binary/boolean discrete classification. Accuracy score: `0.7619926199261993`. The confusion matrix present the results for each class

- Besides the listed attempts, it was also tried ``Label Propagation``, ``Girvan-Newman``, and ``K-Core``algorithms to
  create communities on the citation graph. However, without successes

