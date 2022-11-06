# SUBMIT VERSIONING
# 1: Linear Regression -> 2563315756838200.00000
# 2: Linear Regression -> 256776187267913.00000
# 3: Random Forest Regression  -> 26602.59778
# 4: Random Forest Regression  -> 26235.04907
# 4: Random Forest Regression  -> 26103.75951
# 5: Random Forest Regression  -> 26018.75509
# 5: Random Forest Regression  -> 26018.75509 🎉

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

import config

create_chart = False
config.submition_scoring(False)

# Read data
df_train = pd.read_csv("input_dataset/train.csv")
df_test = pd.read_csv("input_dataset/test.csv")

print("df_train.shape: {}".format(df_train.shape))
print("df_test.shape: {}".format(df_test.shape))


# Preprocess data
def preprocess_data(type, df, train_enc=None):
    # Get percentage of missing values
    percentage_missing_values = df.isnull().sum() / len(df) * 100
    config.generate_chart(
        create_chart,
        "Percentage of missing values",
        percentage_missing_values,
        percentage_missing_values.index,
        "Percentage",
        "Features"
    )

    # get only features with missing values
    tolerance = 70
    features_with_missing_values = percentage_missing_values[percentage_missing_values > tolerance]
    # Drop features with more than 70% missing values
    df.drop(features_with_missing_values.index, axis=1, inplace=True)
    # print("preprocess_data -> after drop missing values : ", len(df.columns))
    config.generate_chart(
        create_chart,
        "Features with more than {}% missing values".format(tolerance),
        features_with_missing_values.index,
        features_with_missing_values,
        "Features",
        "Percentage of missing values"
    )

    # Fill missing values with mean value
    for feature in df.columns:
        if df[feature].isnull().sum() > 0:
            # replace missing values with mean
            if df[feature].dtype == "object":
                df[feature].fillna(df[feature].mode()[0], inplace=True)
            else:
                df[feature].fillna(df[feature].median() , inplace=True)

    # get all outliers for each feature TODO: improve this
    for feature in df.columns:
        if df[feature].dtype != "object":
            # boxplot
            config.generate_barplot(False, df[feature], feature)
            # fig = px.histogram(df, x=df[feature], marginal="box")
            # fig.show()

            # get outliers
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # remplace outliers with mean
            #df[feature] = df[feature].apply(lambda x: df[feature].mean() if x < lower_bound or x > upper_bound else x)

            # remplace outliers with median
            df[feature] = df[feature].apply(lambda x: df[feature].median() if x < lower_bound or x > upper_bound else x)

            # replace outliers with k nearest neighbors
            # from sklearn.impute import KNNImputer
            # imputer = KNNImputer(n_neighbors=2)
            # df[feature] = imputer.fit_transform(df[feature].values.reshape(-1, 1))

            config.generate_barplot(create_chart, df[feature], feature)

    # Create new features
    # df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    # df["Total_sqr_footage"] = (df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"])
    # df["Total_Bathrooms"] = (df["FullBath"] + (0.5 * df["HalfBath"]) + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]))
    # df["Total_porch_sf"] = (df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
    # df["haspool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    # df["has2ndfloor"] = df["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    # df["hasgarage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    # df["hasbsmt"] = df["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    # df["hasfireplace"] = df["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

    # One hot encoding the categorical columns in training set
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Get categorical columns -> Deprecated command
    categorical_columns = df.select_dtypes(include=['object']).columns

    if type == "train":
        train_enc = ohe.fit(df[categorical_columns])
        train_enc_transformed = train_enc.transform(df[categorical_columns])
        final_df = pd.DataFrame(train_enc_transformed)
    if type == "test":
        test_enc = train_enc.transform(df[categorical_columns])
        final_df = pd.DataFrame(test_enc)

    return final_df, train_enc


# Train Model
def train_model(modelType, df_x, df_y):
    model = None
    if modelType == "linear":
        model = LinearRegression()
        model.fit(df_x, df_train_y)
    elif modelType == "random_forest":
        pipeline = Pipeline([('model', RandomForestRegressor())])

        random_forest_param_grid = {
            "model__n_estimators": [200],
            "model__max_leaf_nodes": [50],
            "model__max_depth": [50],
            "model__min_samples_split": [10, 20],
            "model__min_samples_leaf": [10, 25, 35, 45, 55, 60, 100, 200],
            "model__max_features": [10, 25, 35, 45, 55, 60, 100, 200],
            "model__bootstrap": [True, False]
        }

        #random_forest_grid = GridSearchCV(pipeline, cv=5, param_grid=random_forest_param_grid, n_jobs=-1)
        #random_forest_grid.fit(df_x, df_y)
        #print(random_forest_grid.best_estimator_)

        model = RandomForestRegressor(bootstrap=False, max_depth=50,
                                       max_features=100, max_leaf_nodes=50,
                                       min_samples_leaf=10,
                                       min_samples_split=10,
                                       n_estimators=200)
        model.fit(df_x, df_train_y)
    elif modelType == "BaggingRegressor":
        # try bagging regressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        # grid search for bagging regressor
        bagging_regressor_param_grid = {
            "n_estimators": [10, 25, 35, 45, 55, 60, 100, 200],
            "max_samples": [10, 25, 35, 45, 55, 60],
            "max_features": [10, 25, 35, 45, 55, 60],
            "bootstrap": [True, False],
        }
        #bagging_regressor_grid = GridSearchCV(BaggingRegressor(), cv=5, param_grid=bagging_regressor_param_grid, n_jobs=-1)
        #bagging_regressor_grid.fit(df_x, df_y)
        #print(bagging_regressor_grid.best_estimator_)

        bagging_regressor = BaggingRegressor(
            DecisionTreeRegressor(),
            n_estimators=100,
            max_samples=60,
            max_features=60,
            bootstrap=False,
            n_jobs=-1,
        )
        bagging_regressor.fit(df_x, df_train_y)
        model = bagging_regressor
    elif modelType == "HalvingGridSearch":
        # HalvingRandomSearchCV
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingGridSearchCV

        X, y = load_iris(return_X_y=True)
        clf = RandomForestClassifier(random_state=0)

        param_grid = {
            "max_depth": [3,5,20, None],
            "max_features": [1, 3, 10],
            "min_samples_split": [2, 3, 10],
            "min_samples_leaf": [1, 3, 10],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }
        # search = HalvingGridSearchCV(
        #     clf,
        #     param_grid,
        #     resource='n_estimators',
        #     max_resources=10,
        #     random_state=0).fit(X, y)

        # print(search.best_params_)

        # train model RandomForestClassifier
        model = RandomForestClassifier(random_state=0, bootstrap=False, criterion='gini', max_depth=3,
                                       min_samples_leaf=1, min_samples_split=5, n_estimators=100, max_features=10, )
        model.fit(df_x, df_train_y)
    elif modelType == "GradientBoostingRegressor":
        # try gradient boosting
        from sklearn.ensemble import GradientBoostingRegressor
        # grid search for gradient boosting
        gradient_boosting_param_grid = {
            "n_estimators": [10, 25, 35, 45, 55, 60, 100],
            "max_depth": [10, 25, 35, 45, 55],
            "min_samples_split": [10, 25, 35, 45, 55],
            "min_samples_leaf": [10, 25, 35, 45, 55, 60],
            "max_features": [10, 25, 35, 45, 55, 60],
            "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "loss": ["ls", "lad", "huber", "quantile"]
        }
        #gradient_boosting_grid = GridSearchCV(GradientBoostingRegressor(), cv=5, param_grid=gradient_boosting_param_grid, n_jobs=-1)
        #gradient_boosting_grid.fit(df_x, df_y)
        #print(gradient_boosting_grid.best_estimator_)

        model = GradientBoostingRegressor(learning_rate=0.1, loss='ls', max_depth=10, max_features=10, min_samples_leaf=10)

    else:
        print("train_model -> modelType not found")

    if model is not None:
        model_score = model.score(df_x, df_y)
        print("Model score: {}".format(model_score))
        # save model
        print("Your model was successfully saved!")

        import datetime
        now = datetime.datetime.now()
        joblib.dump(model, './saved_model/model_{}.pkl'.format(now.strftime("%Y-%m-%d_%H-%M")))
        return model


# Predict
def predict(model, df):
    predictions = model.predict(df)
    return predictions


#######
# TRAINING
#######
# Split dataset into independent & dependent datasets
df_train_x = df_train.drop('SalePrice', axis=1)
df_train_y = df_train.SalePrice

# Preprocess train data
df_train_x_preprocessed, onehoencodertrained = preprocess_data('train', df_train_x)

# Train model
linear_model_trained = train_model('random_forest', df_train_x_preprocessed, df_train_y)

#######
# PREDICTION
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
#######
# Preprocess test data
df_test_preprocessed, test = preprocess_data('test', df_test, onehoencodertrained)
prediction = linear_model_trained.predict(df_test_preprocessed)

# print root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(df_train_y, linear_model_trained.predict(df_train_x_preprocessed)))
print("Root mean squared error: {}".format(rms))

# Save to csv
config.generate_submission(df_test, prediction)
print(prediction)
