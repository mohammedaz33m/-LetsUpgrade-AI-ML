# LetsUpgrade-AI-ML
An Introduction to Artificial Intelligence &amp; Machine Learning

## Artificial Intelligence & Machine Learning | Batch 1-ML Revision Day 1 Overview
Agenda :
Decision Tree
Overfitting and Underfitting
Entropy
Building Decision Tree
Confusion Matrix
Accuracy and Precision 
Decision Tree:

The basis of decision tree is if else condition in python.
In the given example we choose the suggestion of app based on gender and occupation .
It is basically building a tree based on previous data and using it for predictions.
In case on continuous data we draw the tree using conditional operators .
Also known as Classification and Regression Tree -CART
Geometric Intuition:

We will see how our machine will read the data.
We will draw lines based on conditional operator and create different blocks of data to classify different output .
This is how machine will classify the data .
Conclusion Decision Tree:

Decision tree are nothing but fiant structure of nested if else condition.
Decision tree use hyperplanes which run parallel to a axis and cut the data set.
Pros of Decision Tree :

It is intuitive and easy to understand.
Minimal dara preparation required .
The cost of using the tree for inference is log in the number of data points used to train the data .
Overfitting and Underfitting :

In the example we have a set of question and answer bank given for a test .
We have 3 cases :
One students has worked hard :Sincere
One has done nothing :Un sincere
One has Mugged up things :Mug
In the case of of mugged up student he will answer the question if it asked straightaway but if we twist it a bit he will not be able to answer it .
So when the we twist the data we had bad outcome so this is overfitted data .
In case of the un sincere student how ever we put the question student wont be able to answer this is underfitting as he wont be able to preict anything.
In case of the hardworking student who has understood the answer and studied conceptually if we directly ask the same question or twist it the student will be able to answer the question to a good extent and this is Good fit.
We require a data which has low bias and low variance .
Overfitting:- High Variance
Underfitting :- High Bias
Goodfit:-Low bias ,Low variance.
Cons of Decision tree:

Overfitting.
Prone to errors for imbalanced datasets.
Akinator Game:

Based on if else condition.
This game will keep on asking questions and based on the answer will go through the decision tree and based on our options it will predict about what we were thinking .
This is a classical example where the app will travel down the nodes and in the end where the tree ends we get the answer and this works 99% correctly .
Entropy :

Entropy is the measure of disorder or measure of purity/impurity .
It is also called as degree of randomness.
In case of H20 .
Ice has lowest entropy
Water moderate
Vapours Highest
Building Decision Tree :

We have a dataset with age,competition,type and profit features.
We will find the information gain by the formula.
We have a balanced dataset with 50-50 split of the output .
We will find entropy of all the features .
Entropy is information gain multiplied by probability .
We will check entropy and based on the highest entropy we will choose the Head node in our example we have age as the head node .
We will split the age into types and then the feature with second highest entropy is choosen and so on.
Confusion Matrix :

In the outcome we make a matrix with predicted on x axis and actual on y axis .
In our present scenario of corona testing.
Case 1 : I dont have covid doctor predicted I dont have it :-True negative
Case 2 : I dont have covid doctor predicted I have covid:-False positive
Case 3:I have Covid and doctor preditced I dont have covid :False negative
Case 4 :I have covid and doctor predicted I have covid :True Positive
Accuracy and Precision :

Accuracy is calculated by total number of correctly identified cases by total number of cases .
F1 score takes in consideration precision score and recall.
Precision = TP/(TP+FP).
If dataset it balanced we take accuracy score and if dataset is imbalanced we take F1 Score.

Written by,
@Shoieb Shaikh 


## Artificial Intelligence & Machine Learning | Batch 1-ML Revision Day 3 Overview
Agenda :
Naive Bayes Theorem
The math behind Naive Bayes Theorem
Applications of Naive Bayes Theorem
Implementation of Naive Bayes Theorem
Principal Component Analysis
Implementation of Principal Component Analysis
Working of Naive Bayes Algorithm :

We have a dataset with weather and play as features.
Based on the weather it is decided if they players will play or not.
We will make a frequency table based on the weather if the players have played or not.
We will then compute the probability in different scenarios .
This will be the likelihood table
Prediction:

We have a statement saying “Players will play if the weather is sunny “.
We will calculate :P(Yes|Sunny) Which is the probability of player playing given the weather is sunny .
We see that in our case the probability of the player playing is high in Sunny weather .
Model interpretability is knowing how the model is working and making prediction.
Having knowledge of the way model is predicting is very important .
Where is Naive Bayes Classifier used:

Real time prediction
Text Classification
Recommendation System
Principal Component Analysis:

We use PCA when we have curse of dimensionality .
We prefer removing multi col linearity in other algo but in case of PCA it is compulsory to remove multi-col linearity .
For example we assume a dataset with 5 features .
PCA takes the first feature and find its relation to other columns.
That is,it is trying to find the variance between the features.
So PCA will find this for all the features and makw X1’,X2’ and so on .
Now X1’,X2’ and so on are not related to each other.
X1’ is basically how X1 varies from other features.
We will find out principal component in descending orders.
So we take all the columns find the variance and then we totally transform all the columns then put the descending order of variance or called as principal components.
While working on data multiple dimensions can create a bottle neck so to get better results we use PCA to reduce it to 2D or 3D data .
Large number of features affect the training and accuracy of the model adversely .
To reduce that we can merge the features that are co related .
One more way is to remove some features that are not related by deciding which features is have the most effect on the data.
The feature that causes the highest the maximum variance is said to the first principal component .
Principal Components have no co relation between them.
Normalization of Features :

It is important that we need to normalize the dataset before working on it .
Based on the unit of the records there must be high variance and the prediction will be in favour of high valued data.
Also the data should always be in numerical format .
In PCA we create weighted features based on the original features that and the weighted averages have no relation between them.








# Day 33 Agenda :
Hands on Random Forest Ensemble
Hands on Extra Tree Ensemble
Overfitting and Underfitting
Hands on Ada Boost
Hands on Gradient Boosting
Random Forest Ensemble:

``from sklearn.ensemble import RandomForestClassifier()
rf=RandomForestClassifier(n_estimators=10)``

Training And testing :

``rf.fit(x_train,y_train)
rf_ypred=rf.predict(x_test)
print(“Random Forest Acc_Score”,accuracy_score(y_test,rf_ypred)``

Fine tune the hyper-parameter of Random Forest:

``rf1=RandomForestClassifier(n_estimators=8,max_depth=5,max_feature=6,random_state=2,criterion=’ginni’,verbose=2) #Verbose shows the internal steps in graphical way.
rf1.fit(x_train,y_train)
rf1_ypred=rf1.predict(x_test)
print(“Random Forest 1 Acc_Score”,accuracy_score(y_test,rf1_ypred)``

So we see fine tuning has a very important role in accuracy of the model .
Random Forest does not work on the base but has its own method like other algorithms.
Changing Criterion to Entropy :

``rf2=RandomForestClassifier(n_estimators=8,max_depth=5,max_feature=6,random_state=2,criterion=’entropy’,verbose=2)#Verbose shows the internal steps in graphical way.
rf2.fit(x_train,y_train)
rf1_ypred=rf2.predict(x_test)
print(“Random Forest 2 Acc_Score”,accuracy_score(y_test,rf2_ypred)``

Extra Tree Ensemble:

``from sklearn.ensemble import ExtraTreeClassifier
et=ExtraTreeClassifier()n_estimators=8)
et.fit(x_train,y_train)
rt_ypred=et.predict(x_test)
print(“Accuracy score “,accuracy_score(y_test,et_ypred)``

We dont have a base model here as well .
As it is a extension of Random Forest .
Overfitting and Underfitting:

``print(“Random forest training score”,rf.score(x_train,y_train)
print(“Random forest testing score “,rf.score(x_test,y_test))``

Here we can see that we have almost 100% accuracy but at the time of testing we get low score so this model is overfitted.
The reason could be high variance or it might have many outlier.
This can also be due to unbalanced data .
We can also normalise the data .
Hands on Ada Boost over Log Reg:

``from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(base_estimator=log_reg)
ada.fit(x_train,y_train)
ada_ypred-ada.predict(x_test)
print(accuracy_score(y_test,ada_ypred))
print(“Base estimator logreg’,accuracy_score(y_test,ada_ypred))``

Ada Boost over KNN:

``from aklearn.ensemble import AdaBoostClassifier
ada_knn=AdaBoostClassifier(base_estimator=knn_model,n_estimators=10)
ada_knn.fit(x_train,y_train)
adak_ypred-ada.predict(x_test)
print(accuracy_score(y_test,adak_ypred))
print(“Base estimator knn’,accuracy_score(y_test,adak_ypred))``

Conclusion Ada Boost Doesnt Work on KNN
Gradient Boosting Over Log Reg:

``from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(n_estimator=10)
gb.fit(x_train,y_train)
gb_ypred=gb.predict(x_test)
print(accuracy_score(y_test,gb_ypred)``


# Day 30 Agenda :

Hands on K-Means using Attrition Data set

Principal Componeny Analysis V/s Linear Discriminant Analysis
Hands on LDA in Python
Hands on PCA in Python
Hands on K-Means using Attrition Data set :

``import numpy as np
impost pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(“Attrition.csv”)
col_list=[‘Age’,’education’,’TotalWorkingyears’.’’yearsCireentRole’]
df=data[data.columns[data.columns.isiin(col_list)]]
df.head()
from sklearn.cluster import Kmeans
clust=KMeans(n_clausters=3)
clust.fit(df)
from sklearn.metrics import silhoutte_score
labels=clust.labels_``

silhoutte_score(df,labels) :
Range of it is -1 to +1.
Negative score is bad .
0 means the data points have over lapped .
.71 to 1 is strong
.51 to .7 acceptable
.26 to .5 is weak
Less than .25 is worst .

Plotting L-Bow Curve:
``k=list(range(1,11))
wcss=[]
for i in k
kmean=KMeans(n_clusters=i,random_state=1,init=’k-mean++’)
kmeans.fit(df)
labels=kmeans.labels_
wcss.append(kmeans.inertia_)
plt.plot(k,wcss,color=’n’)
plt..xlabel(“No. Of clusters (k)”)
plt.ylabel(“Within cluster sum of squared errors”)
plt.show``
Looking at the curve we see the best value for k is 2 .

Checking no of values in each cluster
``cluster=KMeans(n_cluster=i,random_state=1)
df[‘cluster’]=cluster.fit_predict(df)
df.cluster.value_counts().sort_index().plot(kind=’bar’.color=’n’)
Plotting for silhouette value:
for i in k
kmean=KMeans(n_clusters=i,random_state=1,init=’k-mean++’)
kmeans.fit(df)
labels=kmeans.labels_
ys.append(kmeans.inertia_)
plt.plot(k,ys,color=’n’)
plt..xlabel(“No. Of clusters (k)”)
plt.ylabel(“Silhouette Score”)
plt.show``

Principal Componeny Analysis V/s Linear Discriminant Analysis :

PCA and LDA both are feature selection technique .
In a specific data set we have to decide the the features that have the most impact on the dataset.
So to determine these important features we use these methods.
Feature Selection Techique
Supervised Learning(Target Based Problem)	Unsupervised Learning (Target Less)
LDA	PCA
Hands on LDA in Python :

``from sklean.discriminant_analysis import LinearDiscriminantAnanlysis as LDA
lda= LinearDiscriminantAnanlysis(n_componenets=2)
lda.fit_transform(x,y)``

Hands on PCA in Python:

``from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca=sklearnPCA(n_component=2)
x_pca=sklearn_pca.fit_transform(x)``



## Artificial Intelligence & Machine Learning | Batch 1-Project 2 -Time Series Forecasting Overview
Agenda :
Time Series 
Importance of Time Series 
Smoothing Methods
Project on Time Series with Airlines dataset
ARIMA Model 
Time Series :

It is an sequence of values of a variable at equal spaced time internal .
It is a series of datapoints ordered in time .
Time series analysis is a statistical technique that deals with time series data ,or trend aalysis.Time series data means that data is in a series of particular periods or intervals.
We refer today as T ,yesterday as T-1 day before yesterday as T-2 and henceforth .And T+1 for tomorrow and so on .
The data is considered in 3 types:
Time series data : A set of observation on the values that a variable take different times .
Cross-Sectional data: Data of one or more variables,collected at the same point in tie
Pooled data: A combination of time series data and cross-sectional data .
We dont take in consideration seasonal data or stats in time series .
We drop trends also in time series because it get vulnerabilities .
After dropping these factors we can do time series forecasting .
We can have different types of time interval .
Here have only 2 variable time and value.
Importance of Time Series :

Time series is very important to solve a lot of problems in the bussiness.
Based on time we create a lot of data .
We can use this to predict future operations.
Time series components:

It can be described in terms of 4 basic classes of components:
Trend:It is a long term direction of a time series .It exist in long term increase or decrease in the data.It does not have to be linear ,sometimes we will refer to a trend “changing direction”when it might go from an increasing trend to a decreasing trend .
Seasonal:It is a regular pattern of variability within certain time periods such as year.
Cyclical:Any regular pattern of sequences of values above and below the trend .
Irregular:Cannot  be defined it can change without a pattern .
Smoothing Methods:

It removes the random variations and shows trend and cyclic components.
When a time series contains a large amount of noise ,it can be difficult to visualize any underlying trend.
There are 2 methods
ARIMA Model:

Auto Regressive(AR) Integrated(I) Moving Average(MA)
ACF &PACF :

Auto Correlation Factor and Partial Auto Correlation Factor 
For practical implementation check Link section in LMS.
