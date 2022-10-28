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
def preprocess_data(type, df, train_enc=None):
    print("-----" * 10)
    print(df.shape)
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

    # One hot encoding the categorical columns in training set
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Get categorical columns -> Deprecated command
    categorical_columns = df.select_dtypes(include=['object']).columns
    print("categorical_columns: {}".format(len(categorical_columns)))
    print(categorical_columns)

    if type == "train":
        train_enc = ohe.fit(df[categorical_columns])
        final_df = pd.DataFrame(train_enc.transform(df[categorical_columns]))

    if type == "test":
        test_enc = train_enc.transform(df[categorical_columns])
        final_df = pd.DataFrame(test_enc)

    # Numerical features
    # df_train_x_numerical = pd.get_dummies(df)
    # print("preprocess_data -> after dummies : ", len(df_train_x_numerical.columns))

    # Fill missing values -> TODO : instead of filling with 0, fill with median
    # df_train_x_numerical.fillna(0, inplace=True)

    # save to csv [debug]
    # df_train_x_numerical.to_csv("df_test_x_numerical.csv", index=False)

    # print("preprocess_data -> after filna : ", len(df_train_x_numerical.columns))

    # df = df_train_x_numerical
    # print("-----" * 10)

    return final_df, train_enc


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
df_train_x_preprocessed, onehoencodertrained = preprocess_data('train', df_train_x)

# Train model
model_trained = train_linear_model(df_train_x_preprocessed, df_train_y)

#######
# PREDICTION
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
#######
# Preprocess test data
df_test_preprocessed, test = preprocess_data('test', df_test, onehoencodertrained)
# print(len(df_test_preprocessed.columns))
prediction = model_trained.predict(df_test_preprocessed)

# Save to csv
df_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': prediction})
df_submission.to_csv('sample_submission.csv', index=False)

print(prediction)
