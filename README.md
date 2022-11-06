# Kaggle Competition House Pricing

` Paul Bouvignies - M1 EXPERT DEVELOPMENT WEB` <br>
` Elective : Mathématique fondamentale et machine learning`
<br>
<br>
> ## Table of contents
> 1. [Data preparation](#introduction) 
>     1. [Drop of columns with too many missing values](#df1)
>     2. [Replace missing values ](#df2)
>     3. [Get outliers and replace values ](#df3)
>     4. [Transform features by numerical categories](#df4)
> 2. [Model training ](#ml)
>     1. [Linear Regressor](#ml1)
>     2. [Random Forest Regressor](#ml2)
>     3. [BaggingRegressor - DecisionTreeRegressor](#ml3)
>     4. [HalvingGridSearch](#ml4)
>     5. [GradientBoostingRegressor](#ml5)
> 3. [Kaggle submission](#kg)

## 1. Data preparation <a name="introduction"></a>
### i. Drop of columns with too many missing values <a name="df1"></a>
<img alt="screenshot" src="md_ressources/screenshot_1.png" width="50%"/><img alt="screenshot" src="md_ressources/screenshot_2.png" width="50%"/>
The first step was to look at the percentage of missing values in changes features. 
If this percentage is over 70% I choose to delete the feature 

### ii. Replace missing values <a name="df2"></a>
To strengthen the dataset, I replace all missing values by the mean of the features.
I also try to replace the missing values by the median but the result is not better.

### iii. Get outliers and replace values <a name="df3"></a>
<img alt="screenshot" src="md_ressources/screenshot_3.png" width="50%"/><img alt="screenshot" src="md_ressources/screenshot_4.png" width="50%"/>
To refine the dataset I replace the exrtemes values by the mean of the feature 

### iv. Transform features by numerical categories <a name="df4"></a>
To allow the training of the model, I convert the features into numerical values that can be computed 

## 2. Model training <a name="ml"></a>
### i. Linear Regressor <a name="ml1"></a>
### ii. Random Forest Regressor <a name="ml2"></a>
### iii. BaggingRegressor - DecisionTreeRegressor <a name="ml3"></a>
### iv. HalvingGridSearch <a name="ml4"></a>
### v. GradientBoostingRegressor  <a name="ml5"></a>
after several tests, I choose the Random Forest Regressor model to make my submission on Kaggle.
Even if other model are suposed to be better, I choose this one because with my data preparation, the model is the most efficient.

### 3. Kaggle submission <a name="kg"></a>
From all my submissions, the best score is 26027.97368. That's not a good score but it's the best I can do with my data preparation and my model.
