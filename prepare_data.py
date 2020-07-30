#!/usr/bin/python3

"""
Do the model-independent data prep
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    plot_precision_recall_curve,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

sns.set()

# This checks if the notebook is executed on Kaggle or on your local machine and
# acts accordingly with filenames.
fname_train = "train.csv"
fname_test = "test.csv"
fdir_out = "./"

try:
    os.environ["KAGGLE_DATA_PROXY_TOKEN"]
except KeyError:
    pass
else:
    dirname = "/kaggle/input/titanic/"
    fname_train = dirname + fname_train
    fname_test = dirname + fname_test
    fdir_out = "/kaggle/woking/"


titanic = pd.read_csv(fname_train)
titanic_orig = titanic


titanic.loc[61, "Embarked"] = "S"
titanic.loc[829, "Embarked"] = "S"


class CabinLetterOnly(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, index=titanic_orig.index, columns=titanic_orig.columns)
        s_cabin_letters = df.Cabin.str.extract("^([A-Z]).*", expand=False)
        return np.c_[df.to_numpy(), s_cabin_letters]


cabin_letter_pip = Pipeline([("cabin_letter_pip", CabinLetterOnly())])

titanic = pd.DataFrame(
    cabin_letter_pip.fit_transform(titanic_orig),
    index=titanic_orig.index,
    columns=titanic_orig.columns.union(pd.Index(["Cabin_letter"]), sort=False),
)


titanic.loc[(titanic.Pclass == 1) & titanic.Cabin_letter.isna(), "Cabin_letter"] = "C"
titanic.loc[(titanic.Pclass == 2) & titanic.Cabin_letter.isna(), "Cabin_letter"] = "F"
titanic.loc[(titanic.Pclass == 3) & titanic.Cabin_letter.isna(), "Cabin_letter"] = "G"

titanic["Family_size"] = titanic_orig.SibSp + titanic_orig.Parch

s_ticket_group = titanic.groupby("Ticket").size().rename("Ticket_group_size")
titanic = titanic.join(s_ticket_group, on="Ticket", sort=False)

titanic["Fare_p_person"] = pd.Series(np.nan)
titanic.loc[titanic.Ticket_group_size == 1, "Fare_p_person"] = titanic.loc[
    titanic.Ticket_group_size == 1, "Fare"
]
titanic.loc[(titanic.Ticket_group_size > 1), "Fare_p_person"] = titanic.loc[
    (titanic.Ticket_group_size > 1), "Fare"
] / (titanic.loc[(titanic.Ticket_group_size > 0), "Ticket_group_size"])

age_round = titanic_orig.Age.dropna().apply(np.floor)
titanic = titanic.join(age_round, lsuffix="_x", sort=False)
titanic.drop("Age_x", axis=1, inplace=True)


knnimputer = KNNImputer(n_neighbors=5, missing_values=np.nan)

mask = titanic.Age.notna() & ((titanic.Fare_p_person < 12) | (titanic.Age < 12))
df_children_or_low_fare = titanic.loc[mask, ["Family_size", "Fare_p_person", "Age"]]
knnimputer.fit(df_children_or_low_fare)
missing_age_low_fare = titanic.loc[(titanic.Fare_p_person < 12) & titanic.Age.isna()][
    ["Family_size", "Fare_p_person", "Age"]
]
imp = knnimputer.transform(missing_age_low_fare)
titanic.loc[(titanic.Fare_p_person < 12) & titanic.Age.isna(), "Age"] = imp[:, 2]

mask = titanic.Age.notna() & (titanic.Fare_p_person >= 12)
df_children_or_low_fare = titanic.loc[mask, ["Family_size", "Fare_p_person", "Age"]]
knnimputer.fit(df_children_or_low_fare)
missing_age_usual_fare = titanic.loc[
    (titanic.Fare_p_person >= 12) & titanic.Age.isna()
][["Family_size", "Fare_p_person", "Age"]]
imp = knnimputer.transform(missing_age_usual_fare)
titanic.loc[(titanic.Fare_p_person >= 12) & titanic.Age.isna(), "Age"] = imp[:, 2]

titanic["Fare_cat"] = pd.cut(
    titanic.Fare_p_person, bins=[0, 13, 30, 513], include_lowest=True
)

titanic["Age_cat"] = pd.cut(titanic["Age"].astype(int), bins=[-1, 20, 40, 60, 90])


sex_pip = Pipeline(
    [("one_hot", OneHotEncoder(categories=[pd.Series.unique(titanic.Sex)]))]
)

fare_cat_pip = Pipeline(
    [("fare_std", OrdinalEncoder(categories=[pd.Series.unique(titanic.Fare_cat)]))]
)

pclass_pip = Pipeline([("pclass_std", StandardScaler())])

age_cat_pip = Pipeline(
    [("age_cat_std", OrdinalEncoder(categories=[pd.Series.unique(titanic.Age_cat)]))]
)

fsize_pip = Pipeline([("fsize_std", StandardScaler())])

cabin_pip = Pipeline(
    [("cabin_1hot", OneHotEncoder(categories=[pd.Series.unique(titanic.Cabin_letter)]))]
)

embarked_pip = Pipeline(
    [("one_hot", OneHotEncoder(categories=[pd.Series.unique(titanic.Embarked)]))]
)

attr_pip = ColumnTransformer(
    [
        ("pclass", pclass_pip, ["Pclass"]),
        ("sex", sex_pip, ["Sex"]),
        ("fare_cat", fare_cat_pip, ["Fare_cat"]),
        ("age_cat", age_cat_pip, ["Age_cat"]),
        ("fsize", fsize_pip, ["Family_size"]),
        ("cabin", cabin_pip, ["Cabin_letter"]),
        ("embarked", embarked_pip, ["Embarked"]),
    ],
    remainder="passthrough",
)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_ix, test_ix in split.split(titanic, titanic["Survived"]):
    strat_train_all = titanic.loc[train_ix]
    strat_devtest_all = titanic.loc[test_ix]

titanic_labels = titanic["Survived"].astype(int)
strat_train_labels = strat_train_all["Survived"].astype(int)
strat_devtest_labels = strat_devtest_all["Survived"].astype(int)
strat_train_all.drop(["Survived"], inplace=True, axis=1)
strat_devtest_all.drop(["Survived"], inplace=True, axis=1)

# key columns are included, columns in value lists are dropped
drop_col_lists = {
    # "Sex, Pclass, Fare_cat, Family_size, Age_cat, Embarked, Cabin_letter": [
    #     "PassengerId",
    #     "Cabin",
    #     "SibSp",
    #     "Parch",
    #     "Ticket",
    #     "Name",
    #     "Fare",
    #     "Age",
    #     "Ticket_group_size",
    #     "Fare_p_person",
    # ],
    # "Sex, Pclass, Fare_cat, Family_size, Age_cat, Embarked": [
    #     "PassengerId",
    #     "Cabin",
    #     "SibSp",
    #     "Parch",
    #     "Ticket",
    #     "Name",
    #     "Fare",
    #     "Age",
    #     "Ticket_group_size",
    #     "Fare_p_person",
    #     "Cabin_letter",
    # ],
    "Sex, Pclass, Fare_cat, Family_size, Age_cat": [
        "PassengerId",
        "Cabin",
        "SibSp",
        "Parch",
        "Ticket",
        "Name",
        "Fare",
        "Age",
        "Ticket_group_size",
        "Fare_p_person",
        "Cabin_letter",
        "Embarked",
    ],
    # "Sex, Pclass, Fare_cat, Family_size": [
    #     "PassengerId",
    #     "Cabin",
    #     "SibSp",
    #     "Parch",
    #     "Ticket",
    #     "Name",
    #     "Fare",
    #     "Age",
    #     "Ticket_group_size",
    #     "Fare_p_person",
    #     "Cabin_letter",
    #     "Embarked",
    #     "Age_cat",
    # ],
    # "Sex, Pclass, Fare_cat": [
    #     "PassengerId",
    #     "Cabin",
    #     "SibSp",
    #     "Parch",
    #     "Ticket",
    #     "Name",
    #     "Fare",
    #     "Age",
    #     "Ticket_group_size",
    #     "Fare_p_person",
    #     "Cabin_letter",
    #     "Embarked",
    #     "Age_cat",
    #     "Family_size",
    # ],
}

cols_pipelines = {
    "Sex": sex_pip,
    "Pclass": pclass_pip,
    "Fare_cat": fare_cat_pip,
    "Family_size": fsize_pip,
    "Age_cat": age_cat_pip,
    "Embarked": embarked_pip,
    "Cabin_letter": cabin_pip,
}
