# Kaggle
The projects on Kaggle, using R or Python
### This repository consists of three kaggle topics
 * Ames Housing Price Presiction
 * Titanic Survival Prediction
 * Tripadvisor Reviews Clustering

### (i) Ames Housing Price Presiction
  * 1. 	Background <br>
Purpose: learn 3 different algorithms to do prediction. This project includes data split, data cleaning, data exploration, feature engineering and model building.

  * 2.  data split <br>
Before building the model , we need to divide the data into training and test set. The training part is used for building the predictive model, and the test part is to check the accuracy of that.

  * 3.  data cleaning <br>
Raw data usually has some outliers and missing values which could affect the accuracy of the model. So in this part, we need to deal with them before feature engineering

  * 4.  feature engineering <br>
Before we conduct feature engineering. We should have a quick look at the importance of variables after implemented data cleaning. We pick out the variables we consider as important. 

  * 5.  prepare the data for model <br>
At first, we should drop highly correlated variables. Then we need to normalize the numeric predictors. For factor variables, we need to do the one hot encoding to normalize them, using model.martix() . After that, we need to drop the variables whose levels are less than 10, because less levels will not be that accurate in modeling 

  * 6. logit the sale_price <br>
  Dealing with skewness of response variable can switch the response variables into normal distribution. 

  * 7. build the model <br>
    * XGboost <br>
    XGboost takes the shortest time when building the prediction model. And its accuracy is the best. 
    
    * Lasso
    * GAM


### (ii)	Classification (Titanic data set)
In this project, I used the Titanic data set which has been used on Kaggle (https://www.kaggle.com/broaniki/titanic) as the basis for a competition entitled Titanic: Machine Learning from Disaster. I used classification method to analyze the relationship between the predictors and passengers’ survival probability, and how to choose the most significant ones to build the model. 

  * Note: the steps I conducted below involve data cleaning, building the model, estimation and plotting...

  * 1.Data overview ** <br>
I used the function below to generate train and test set. And then select the °∞survived°± factor to be 1 or 0 to calculate the number in female, male and children who died or survived.
```python
x_train, x_test, y_train, y_test = 
train_test_split(predictors, target, test_size = 0.10, random_state = 0)
```
  * 2.Balanced ** <br>
We have to make sure that the train and test data are balanced.

  * 3.Logistic Regression ** <br>
Logistic regression is also a good algorithm to do prediction. The ways to test the performance of the prediction model on the test data is also the accuracy score and confusion matrix. Then I finally used the code below to generate the data I want for which I chose ["Pclass","Sex","Age","Fare","Embarked"] to be the predictors and then I library the method to build the logistic model and ROC curve. 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
col_n = ["Pclass","Sex","Age","Fare","Embarked"]
x_train = pd.DataFrame(train,columns = col_n)
x_val = pd.DataFrame(test,columns = col_n)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
```
  * 4.Linear Regression ** <br>
For linear Regression, I used the data[°ÆFare°Ø, °ÆAge°Ø]. I do the Age regression based on Fare.
The libraries I used for modeling, measurement and plotting are below.
 
 * 5.Decision Tree ** <br>
After doing the feature engineering, I transfer the numeric features into categorical numeric features. Then I used the code below to library the method I need to build a decision tree,here I choose ["Pclass","Sex"] to be the two attributes for model:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
col_n = ["Pclass","Sex"]
x_train = pd.DataFrame(train,columns = col_n)
y_train = train['Survived']
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
```
The below is how decision tree looks like:<br>
<img src="https://github.com/nicolehhy/Kaggle/blob/master/Picture1.png" width="500" alt="decision tree">


### (iii)	Clustering (Trip Advisor Reviews data set)
In this project, I used Trip Advisor Reviews data set which is from (https:// archive.ics.uci.edu/ml/datasets/Travel+Reviews), which is a web site hosted at the University of California at Irvine (UCI) that contains a large number of data sets which people use for testing data mining methods. I used multiple clustering methods in this part to category the 10 attributes for the clustering model.

  * 1.Data overview ** <br>
I used data.describe() to take a look of the data. There are 980 instances in the whole data set. And there is no missing values in every attributes which is a good start.

  * 2.K-means clustering ** <br>
In this case, I put K into the list which is 3,5,10. There are 980 instances in the whole data set. And there is no missing values in every attributes which is a good start. The results of overall clustering score and silhouette coefficient and the Calinski score, also the clustering 2D plot. I used the code below to load the libraries I need for the process.

  * 3.Agglomerative clustering **  <br>
A In order to do the comparison, K here is also should be one of 3,5,10. This algorithm is different from K-means method. It doesn°Øt have overall score to do estimation. However, it still involves many other ways including the 2D plot to test the performance. I used the code below to load the libraries I need for the process.

  * 4.DBSCAN clustering **  <br>
The difference between the two above and DBSCAN is DBSCAN doesn°Øt need to set the clustering numbers. In this algorithm, eps and min_sample are two important parameters in the model. When eps goes up, the number of categories will be larger. When mini_sample goes down, the number of categories will be bigger. In order to do the comparison, K here is also should be one of 3,5,10. So in this case, I set the parameters specifically so that the clustering numbers would be the same with another two algorithms. 
