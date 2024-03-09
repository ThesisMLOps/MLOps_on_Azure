import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

TARGET_COL = "action_taken"

FEATURE_COLS = [
    "derived_sex", "preapproval", "loan_type", "loan_purpose",
    "lien_status", "reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
    "interest_only_payment", "balloon_payment",	"occupancy_type", "derived_age_above_62",
    "derived_age_below_25",	"derived_race_revisited", "if_co-applicant", "loan_amount",
    "loan_to_value_ratio", "loan_term",	"property_value", "total_units", "income",
    "debt_to_income_ratio"
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # classifier specific arguments
    parser.add_argument('--regressor__n_estimators', type=int, default=100,
                        help='Number of trees')
    parser.add_argument('--regressor__bootstrap', type=int, default=True,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--regressor__max_depth', type=int, default=None,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--regressor__max_features', type=str, default='sqrt',
                        help='Number of features to consider at every split')
    parser.add_argument('--regressor__min_samples_leaf', type=int, default=1,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--regressor__min_samples_split', type=int, default=2,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()

    return args
    
def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_csv((Path(args.train_data) / "train.csv"), index_col=False)    
    #train_data = pd.read_parquet(Path(args.train_data))
    #train_data_mltable = mltable.load(Path(args.train_data))
    #train_data = train_data_mltable.to_pandas_dataframe()

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[FEATURE_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestClassifier(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=None)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    accuracy = accuracy_score(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train accuracy score", accuracy)
    print("train accuracy score ", accuracy)

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""
"""
import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance", "dropoff_latitude", "dropoff_longitude", "passengers", "pickup_latitude",
    "pickup_longitude", "pickup_weekday", "pickup_month", "pickup_monthday", "pickup_hour",
    "pickup_minute", "pickup_second", "dropoff_weekday", "dropoff_month", "dropoff_monthday",
    "dropoff_hour", "dropoff_minute", "dropoff_second"
]

CAT_NOM_COLS = [
    "store_forward", "vendor"
]

CAT_ORD_COLS = [
]


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # classifier specific arguments
    parser.add_argument('--regressor__n_estimators', type=int, default=500,
                        help='Number of trees')
    parser.add_argument('--regressor__bootstrap', type=int, default=1,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--regressor__max_depth', type=int, default=10,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--regressor__max_features', type=str, default='auto',
                        help='Number of features to consider at every split')
    parser.add_argument('--regressor__min_samples_leaf', type=int, default=4,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--regressor__min_samples_split', type=int, default=5,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=0)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    """
