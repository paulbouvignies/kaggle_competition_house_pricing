import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Style for plots
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# Function tha generate chart with data
def generate_chart(enabled=False, title='title', x=0, y=0, x_label="x_label", y_label="y_label"):
    if enabled:
        plt.figure(figsize=(10, 5))
        plt.title(title)
        sns.barplot(y=y, x=x)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


def generate_barplot(enabled, df, feature):
    if enabled:
        sns.boxplot(df)
        plt.title("Boxplot for {}".format(feature))
        plt.show()


def generate_submission(df_test, prediction):
    import datetime
    now = datetime.datetime.now()

    df_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': prediction})
    df_submission.to_csv('./output_dataset/sample_submission.csv', index=False)
    df_submission.to_csv('./output_dataset/sample_submission_{}.csv'.format(now.strftime("%Y-%m-%d_%H-%M")), index=False)
    print("Your submission was successfully saved!")


def submition_scoring(enabled=False):
    score = [
        2563315756838200.00000,
        256776187267913.00000,
        26602.59778
    ]
    if enabled:
        # draw chart with all score values
        generate_chart(enabled=True, title='Score', x=score, y=score, x_label="Score", y_label="Score")


def get_outliers(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
