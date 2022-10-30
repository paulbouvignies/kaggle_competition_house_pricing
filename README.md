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
>     3. [Gradien tBoosting Regressor](#ml3)
> 3. [Kaggle submission](#kg)
> 4. [Conclusion](#conclusion)

## 1. Data preparation <a name="introduction"></a>
### i. Drop of columns with too many missing values <a name="df1"></a>
<img alt="screenshot" src="md_ressources/screenshot_1.png" width="50%"/><img alt="screenshot" src="md_ressources/screenshot_2.png" width="50%"/>
The first step was to look at the percentage of missing values in changes features. 
If this percentage is over 70% I choose to delete the feature 

### ii. Replace missing values <a name="df2"></a>
To strengthen the dataset, I replace all missing values by the mean of the features

### iii. Get outliers and replace values <a name="df3"></a>
<img alt="screenshot" src="md_ressources/screenshot_3.png" width="50%"/><img alt="screenshot" src="md_ressources/screenshot_4.png" width="50%"/>
Ro refine the dataset I replace the exrtemes values by the mean of the feature 

### iv. Transform features by numerical categories <a name="df4"></a>
To allow the training of the model, I convert the features into numerical values that can be computed 

## 2. Model training <a name="ml"></a>
### i. Linear Regressor <a name="ml1"></a>
### ii. Random Forest Regressor <a name="ml2"></a>
### iii. Gradien tBoosting Regressor <a name="ml3"></a>
