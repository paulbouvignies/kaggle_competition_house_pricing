import matplotlib.pyplot as plt
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


def generate_submission(df_test,prediction):
    df_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': prediction})
    df_submission.to_csv('sample_submission.csv', index=False)
    print("Your submission was successfully saved!")