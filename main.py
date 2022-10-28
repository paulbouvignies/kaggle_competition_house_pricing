# SUBMIT VERSIONING
# 1: Linear Regression -> 2563315756838200.00000
# 2: Linear Regression -> 256776187267913.00000
# 3: Random Forest Regression  -> 26602.59778 🎉

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


import config

create_chart = False

# Read data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

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

    # One hot encoding the categorical columns in training set
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Get categorical columns -> Deprecated command
    categorical_columns = df.select_dtypes(include=['object']).columns

    if type == "train":
        train_enc = ohe.fit(df[categorical_columns])
        final_df = pd.DataFrame(train_enc.transform(df[categorical_columns]))

    if type == "test":
        test_enc = train_enc.transform(df[categorical_columns])
        final_df = pd.DataFrame(test_enc)


    return final_df, train_enc


# Train Linear Model
def train_model(modelType,df_x, df_y):
    model = None
    if  modelType == "linear":
        model = LinearRegression()
        model.fit(df_x, df_train_y)
    elif modelType == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(df_x, df_train_y)
    else:
        print("train_model -> modelType not found")

    if model is not None:
        model_score = model.score(df_x, df_y)
        print("Model score: {}".format(model_score))
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
linear_model_trained = train_model('random_forest',df_train_x_preprocessed, df_train_y)


#######
# PREDICTION
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
#######
# Preprocess test data
df_test_preprocessed, test = preprocess_data('test', df_test, onehoencodertrained)
# print(len(df_test_preprocessed.columns))
prediction = linear_model_trained.predict(df_test_preprocessed)

# Save to csv
config.generate_submission(df_test,prediction)
# print(prediction)
