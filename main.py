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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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
                df[feature].fillna(df[feature].mean(), inplace=True)

    # get all outliers for each feature TODO: improve this
    for feature in df.columns:
        if df[feature].dtype != "object":
            # boxplot
            config.generate_barplot(create_chart, df[feature], feature)
            # fig = px.histogram(df, x=df[feature], marginal="box")
            # fig.show()

            # get outliers
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # remplace outliers with mean
            df[feature] = df[feature].apply(lambda x: df[feature].mean() if x < lower_bound or x > upper_bound else x)
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
            "model__n_estimators": [35],
            "model__max_leaf_nodes": [300],
            "model__max_depth": [400],
            "model__min_samples_split": [10],
            "model__min_samples_leaf": [10, 25, 35, 45, 55, 60, 100, 200],
            "model__max_features": [10, 25, 35, 45, 55, 60, 100, 200],
            "model__bootstrap": [True, False]
        }

        # random_forest_grid = GridSearchCV(pipeline, cv=5, param_grid=random_forest_param_grid, n_jobs=-1)
        # random_forest_grid.fit(df_x, df_y)
        # print(random_forest_grid.best_estimator_)

        model = RandomForestRegressor(max_depth=400, max_leaf_nodes=300, min_samples_split=10, n_estimators=35)
        model.fit(df_x, df_train_y)
    elif modelType == "BaggingRegressor":
        # try bagging regressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        bagging_regressor = BaggingRegressor(
            DecisionTreeRegressor(),
            n_estimators=500,
            max_samples=100,
            bootstrap=True,
            n_jobs=-1,
            oob_score=True
        )
        bagging_regressor.fit(df_x, df_train_y)
        model = bagging_regressor
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
linear_model_trained = train_model('BaggingRegressor', df_train_x_preprocessed, df_train_y)

#######
# PREDICTION
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
#######
# Preprocess test data
df_test_preprocessed, test = preprocess_data('test', df_test, onehoencodertrained)
prediction = linear_model_trained.predict(df_test_preprocessed)

# Save to csv
config.generate_submission(df_test, prediction)
print(prediction)
