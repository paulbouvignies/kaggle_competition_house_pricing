import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import config

create_chart = False

# Read data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("df_train.shape: {}".format(df_train.shape))
print("df_test.shape: {}".format(df_test.shape))


# Preprocess data
def preprocess_data(df):
    print("-----" * 10)
    # Get percentage of missing values
    percentage_missing_values = df.isnull().sum() / len(df) * 100;
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
    print(features_with_missing_values.index)
    # Drop features with more than 70% missing values
    df.drop(features_with_missing_values.index, axis=1, inplace=True)
    print("preprocess_data -> after drop missing values : ",len(df.columns))
    config.generate_chart(
        create_chart,
        "Features with more than {}% missing values".format(tolerance),
        features_with_missing_values.index,
        features_with_missing_values,
        "Features",
        "Percentage of missing values"
    )

    # Numerical features
    df_train_x_numerical = pd.get_dummies(df)
    print("preprocess_data -> after dummies : ",len(df_train_x_numerical.columns))

#########
    # Saving the columns in a list
    cols = df_train_x_numerical.columns.tolist()
    cols.fillna(0, inplace=True)
#########

    #df_train_x_numerical.fillna(0, inplace=True)
    # save to csv
    #df_train_x_numerical.to_csv("df_test_x_numerical.csv", index=False)
    print("preprocess_data -> after filna : ",len(df_train_x_numerical.columns))

    df = df_train_x_numerical
    print("-----" * 10)

    return df


# Train Linear Model
def train_linear_model(df_x, df_y):
    model = LinearRegression()
    model.fit(df_x, df_train_y)
    model_score = model.score(df_x, df_y)
    print("Model score: {}".format(model_score))
    return model


# Predict
def predict(model, df):
    predictions = model.predict(df)
    print(predictions)


#######
# TRAINING
#######
# Split dataset into independent & dependent datasets
df_train_x = df_train.drop('SalePrice', axis=1)
df_train_y = df_train.SalePrice

# Preprocess train data
df_train_x_preprocessed = preprocess_data(df_train_x)

# Train model
model_trained = train_linear_model(df_train_x_preprocessed, df_train_y)

#######
# PREDICTION
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
#######
# Preprocess test data
df_test_preprocessed = preprocess_data(df_test)
# print(len(df_test_preprocessed.columns))
model_trained.predict(df_test_preprocessed)