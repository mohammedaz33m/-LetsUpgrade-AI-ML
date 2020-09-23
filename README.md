# LetsUpgrade-AI-ML
An Introduction to Artificial Intelligence &amp; Machine Learning | All agenda written by @shoeb (community co-ordinator)

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





# DAY 23 AGENDA
Practical Issues of decision tree
Overfitting 
Handling Missing Values
Pros  and Cons of Decision tree
Random Forest
Hands-on Decision Tree In Python
Practical Issues of decision tree:
Overfitting or underfitting, missing values, and costs of classification.
These are 3 practical issues we will face while going for a decision tree.
Overfitting :
Overfitting means if we have more independent variables we can have problems in using the decision tree.
We can use the pre-pruning method to solve the problem of overfitting issues.
Random forest is an algorithm which is n-sample of decision tree and using random forest we can identify the important variables for decision tree.
Overfitting means if there are many variables due to which the decision tree grows complex.
We can use pre-pruning to solve this issue and there is 2 techniques used in this method.
The 2 methods are forward pruning and backward pruning.
We use these to stop the tree growing from either width-wise or breadthwise.
Pre-pruning is controlling the tree before tree generation.
We can do pre-pruning with the random forest algorithm.
And post – pruning is done after the generation of the tree.
Post pruning has 2 methods that are subtree replacement and subtree rising 
Subtree replacement is done by checking which node is growing more and analyzing it.
This increases the accuracy of the algorithm.
If the tree is rising more and more we can directly connect the more growing node directly to main node and this is subtree rising method.
Handling Missing Values :
We have to also handle the missing values.
There are 2 methods to it.
In the case of many records, we can delete a particular column.
In the case of less response, we can replace them.
We can take the average and fill the empty cells when the input/independent variable is continuous.
In the case of categorical, we can substitute the empty spaces using most occuring value
Pros  of the Decision tree:
WE can generate rules using a Decision tree.
We can perform classification easily.
A decision tree can handle both continuous and categorical features .
It provides a clear vision of the fields that are most important for classification and prediction.
It is very fast in classification.
Inexpensive to construct.
Easy to interpret 
 Cons Of Decision Tree :
The dependent variable should be categorical only.
Gives lesser accuracy with many class and small data.
Computationally expensive to train.
It is difficult to classify when there are many categories.
Random Forest :
It is an ensemble method of decision tree.
It is a algorithm which is used to find important variables for decision tree.
Used to choose the most important variables.
It is used to find which independent variable is having more depth and can be used to classify the records.
Hands on Decision Tree In Python:
import pandas ,numpy
from sklearn import tree and preprocessing 
preprocessing is used for data processing for example conversion from categorical to continuous data .
titanic_train=pd.read_csv(“name”)#Loading the training data set 
new_age_var=np.where(titanic_train[“Age”].isnull(),32,titanic_train[“Age”]#finding the mean from continuous feature and replace all the missing values with mean.
titanic_train[“Age”]=new_age_var# reasgining 
We can use label binariser to convert categorical to continuous value only when there are 2 categories other wise label encoder .
label_encoder=preprocessing.LabelEncoder() #assigning label encoder to a variable.
encoded_sex=label_encoder.fit_tranform(titanic_train[“sex”]) #Encoding sex to continuous variable .
tree_model=tree.DecisionTreeClasifier() #
tree_model.fit(x=pd.DataFrame(encoded_sex),y=titanic_train[“Survived”])#passing the dependent and independent variable x is independent.Created the model
with open(“Dtree1.dot.”,’w’) as f#visualizing the model 
f=tree.export_graphviz(tree_model,feature_nammes=[“Sex”],out_file=f);#This will write data in the file in the format the computer can understand
Visit webgraphviz.com and use it to generate the graphs.
Here dependent/dv is survived and independent variable is gender .
predictors=pd.DataFrame([encoded_sex,titanic_train[“Pclass”]]).T#Adding IDV 
tree_model.fit(x=predictors,y=titanic_train[“Survived”])
with open(“Dtree.dot”,’w’) as f:
f=tree.export_graphviz(tree_model,feature_names=[“Sex”,”pclass”],out_file=f;#Model with 2 independent variable



# DAY 24 AGENDA

Hands-on Decision Tree in Python
Testing Decision Tree Algorithm
Hands-on Random Forest in Python
Hands-on Decision Tree in Python:
Adding more features or independent variables.
We cannot add them directly so we put it in a DataFrame and add the DataFrame .
predictors=pd.DataFrame([encoded_sex,titanic_train[“Pclass”],titanic_train[“Age”],titanic_train[“Fare”]) #Adding IDV
tree_model=tree.DecisionTreeClassifier(max_depth=8)# Specifying the max depth so the tree doesn’t grow a lot .It is one of the pre-pruning technique.
tree_model.fit(x=predictors,y=titanic_train[“Survived’])
with open(“Dtree3.dot”,’w’)as f:  #Creating an output file with code for the model
f=tree.export_graphviz(tree_model,feature_names=[“Sex”,”Pclass”,”Age”,”fare”],out_file=f);
With more and more independent variable the tree gets very complicated and it gets difficult to classify the record.
To check how reliable it is we can check the model accuracy as well.
tree_model.score(x=predictors,y=titanic_train[“Survived”]) #This will give the percentage accuracy of the model .
In our example, the model is 89% accurate.
To get more accuracy we need to cut on the independent variable and use only the important ones.
To get important variables we can use random forest algorithm.
Testing:
In the test dataset, we do not have the column which has the dependent or the target variable.
So based on the training data set we will predict the test data set .
titanic_test=pd.read_csv(“test.csv”) #Loading the test data set .
new_age_var=np.where(titanic_test[“Age”].isnull(),28,titanic_test[“Age”]) #replacing null values .
titianic_test[“age”]=new_age_var # re-assigning
encoded_sex_test=label_encoder.fit_tranform(titanic_test[“Sex”]) #Encoding gender from categorical to continuous
test_features=pd.dataFrame([encoded_sex_test,titanic_test[“Pclass”],titanic_test[“age”],titanic_test[“Fare”]]).T #Loading features into the DataFrame
test_preds=tree_model.predict(x=test_features)#Loading independent variable
predicted_output =pd.DataFrame({“PassengerId”:titanic_test[‘PassengerId’],’Survived”:test_preds})#Predicting the survived columns
predicted_output.to_csv(“Output.csv”,index=False) #Getting results into a csv format with survived column .
Hands-on Random Forest in Python:
Random forest is an ensemble of the decision tree.
It is used to check the important independent variables.
From sklearn.ensemble import RandomForestClassifier
We will use the training data set to check the most important independent variable.
label_encoder=preprocessing.LabelEncoder()
titanic_train[“Sex”]=label_encoder.fit_transform(titanic_train[“Sex”]) #Encoding gender into continuous type.
titanic_train[“Embarked”]=label_encoder.fit_transform(titanic_train[“Embarked”]) #Encoding embarked  into continuous type.
rf_model=RandomForestClassifier(n_estimators=1000,max_feature=2,oob_score=Tree) #Initializinf Random forest model where estimator tells no of iteration and max feature defines binary split ,oob_score that is out of bag score means we can find accuracy  on every variable.
features=[“Sex”,”Pclass”,”SibSp”,”Age”,”Fare”] #Defining independent variables.
rf_model.fit(x=titanic_train(features),y=titianic_train[“Survived’])
print(“OOB Accuracy”)
print(rf_model.oob_score_); #Gives OOB Accuracy
for feature,imp in zip(features,rf_model.feature_importances_):
print(feature,imp) #displays oob_score for every independent variables.
Here we can decide the more important variable.
Assignment:
Project 1:-Using the titanic data set create a model with the 3 most important independent variable that is gender age and fare to get a dependent variable as survived and perform prediction.
Project 2:-Apply Decision tree and random algorithm on attrition dataset with DV as Attrition and choose the proper IDV using Random forest.
Project 3:- Apply Decision tree and random algorithm on  bank_modelling dataset with DV as a personal loan and find IDV using Random Forest Algo.
In all The Project makes the model test it and also writes down the rules.


# DAY 25 AGENDA

Naive Bayes Theorem
Classification Technique
Hands-On in Python Of Naive Bayes Theorem
Training and Testing
Naive Bayes Theorem :
This is a classification technique.
We use probability to classify the records.
We can apply this theorem only when we have dependent and independent variables categorical.
It is useful for classifying large datasets.
We use probability to classify the records and with proper formula and structure.
In the example dataset, we have a dependent variable is if the company is truthful or not.
The independent variable is the size of the company or the company is legally charged or not.
We will check how many companies are fraud and then out of those we will check according to the size and those fraud ones we have 3 companies with charge yes.
We will find the probability of the companies fraud out of that how many frauds, small in size and charged or not, and then multiply the probability of each.
p(fraud|charges,small) =(4/10)*(1/4)*(3/4)=0.075/(0.075+0.067)=0.53
p(truthful|charges,small)=(1/6)*(4/6)*(6/10)=0.067/(0.075+0.067)=0.471
Looking at data we calculated we can say that if the company charges and it is small we can classify it is a fraud company.
In the next example, we have 14 days of data of the badminton series which has all the weather data and the dependent variable says if the match was played or not.
The Independent variable here is the outlook, temperature, humidity, wind and the dependent variable is if the match was played or not.
We will then look for the number of records and check how many days match was played and in those days how many days the outlook is sunny and further specific details.
We will find the probability for different scenarios.
Finding the probability for different conditions is called the conditional probability.
We have to predict on a day sunny, cool temp, high humidity, and strong wind.
To find the prediction we multiply yes for all conditions and no for all the conditions and we find if yes has a higher probability the match will be played or else it will no be played.
It finds its application in Gmail for spam classification.
Classification Technique
Technique	Dependent Variable	Independent variable	Purpose
Naive Bayes Theorem	Categorical 	Categorical	Classification technique used to classify the records using probability 
K-nearest neighbor(KNN)	Categorical	Categorical and continuous 	Classification technique used to classify record with the help of Euclidean Distance
Support Vector Machine(SVM)	Categorical	Categorical and continuous	Classification technique and used to classify record with the help of Hyperplane and is applicable for over-dimensional data 
Hands-On in Python Of Naive Bayes Theorem :
Import pandas
load the data set 
from sklearn import p reprocessing
from sklearn.cross_validation import train_test_split (if cross-validation is not supported we can use sklearn.model_selection for loading train_test_split function)
from sklearn.naive_bayesimport GaussianNB 
from sklearn.metrics import accuracy_score #Tells the model accuracy.
From sklearn.matrics import confusion_matrix  #Confusion matrix tells us how many reocrds are classified accurately .
le=preprocessing.LabelEncoder()
le.fit(dataset[“Sec”]
print(le.classes_) #Gives out classes
dataset[“Sex’]=le.transform(dataset[“Sex”]) #Coverting categorical to Continuous data
y=dataset[“Survived”] #Defining dependent varible
X=dataset.drop([“Survived”,”PassengerId”],axis=1) #axis=1 defines columns and 0 defines rows
y.count() #Gives the number of record
Independent variable is denoted by X and dependent by y.
Training and Testing :
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)  #Splitting in training and testing dataset
X_train.head()
from sklearn.naive_bayes import *
clf=BernoulliNB()
y_pred-clf.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_pred,normalize=True) #Gives out Accuracy score
confusion_matrix(y_test,y_pred) #checks how many records were classified correctly
Reading Confusion Matrix :
Training dataset is in the column and test or predicted in the rows .
The diagonals are correctly classified.
Other than diagonal others are no classified correctly
Assignment:  
First take pclass , gender, SibSP , ParCh, Embarked as dependent variable one by one and others as the independent variable and find the confusion matrix. Using looping or function how can you minimize the code.



# DAY 26 AGENDA

Random Forest and Decision Tree of Bank Loan Data set.
Random Forest and Decision Tree of Attrition Data set.
Logistic Regression on Bank Loan Data set.
Logistic Regression on  Attrition Data set.
Random Forest of Bank Loan Dataset:
A personal loan is the dependent variable and Others are the independent variables.
Import pandas
dataset=pd.read_excel(“Bankmodelingsheet”)
dataset1=dataset.drop([“ID”,”ZIP CODE’],axis=1) #Removing unwanted columns
dataset2=dataset1.dropna() #Removing null value
dataset3=dataset2.drop_duplicates() #Removing duplicate values
from sklearn.ensemble import RandomForestClassifier 
import numpy 
dataset2[“CCAvg”]=np.round(dataset2[“CCAvg”]) #Rounding up values
rf_model=RandomForestClassifier(n_estimator=1000,max_feature=2,oob_score=True) #Assigning model to variable
feature=[All the columns in brackets separated by columns without nit required values]
rf_model.fit(X=dataset3[features],y=dataset3[“Personal Loan”]) #Defining DV and IDV
print(“OOB Accuracy”)
print(rf_model.oob_score_); #Gives out oob score
for feature,imp in zip(features,rf_model.feature_importances_):
print(feature,imp);
#Gives out importance of all the variables
Choose the variable with the highest importance
Decision Tree Of Bank loan dataset:
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)
predictors=pd.DataFrame([dataset3[“Education”],dataset3[“CCAvg”],dataset3[“Income”]]).T #Defining most important features
tree_model.fit(X=predictors,y=dataset3[“Personal Loan”]) #Model fitment
with open(“Dtree.dot”,’w’) as f:
f=tree.export_graphviz(tree_model,feature_names=[“Education”,”CCAvg”,”Income”],out_file=f);
Plot the tree and read it .
Random Forest of Attrition Dataset:
dataset4=pd.read_csv(“general_data.csv”)
from sklearn import preprocessing 
le=preprocessing.LabelEncoder()
dataset4[“Attrition”]=le.fit_tranform(dataset4[“Attrition”]) #Converting categorical to continuous
dataset4[“Business Travel”]=le.fit_tranform(dataset4[“Business Travel ”]) 
dataset4[“Department”]=le.fit_tranform(dataset4[“Department”])
dataset4[“EducationField”]=le.fit_tranform(dataset4[“EducationField”])
dataset4[“Gender ”]=le.fit_tranform(dataset4[“Gender”])
dataset4[“MaritalStatus”]=le.fit_tranform(dataset4[“MaritalStatus”])
dataset4[“JobRole”]=le.fit_tranform(dataset4[“JobRole”])
dataset5=dataset4.drop([“EmployeCount”,”EmployeeID”,”Over18”,”StandardHours”],axis=1)
from sklearn.ensemble import RandomForestClassifier
 dataset6=dataset5.dropna() #Dropping null
dataset7=dataset6.drop_duplicates() #Dropping Duplicates
rf_model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=true) #Defining model
features=[All the features that are important and excluding the attrition column]
rf_model.fit(X=dataset7[features],y=dataset7[“Attritiom”])
print(“OOB Accuracy “)
print(rf_model.oob_score_); #Gives out the accuracy score
for feature,imp in zip(features,rf_model.feature_importances_):
print(feature,imp); #Gives out the importance level of each variables
 

Decision Tree Of Attrition dataset
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)
predictors=pd.DataFrame([dataset7[“Age”],dataset7[“MonthlyIncome”],dataset7[“WorkingYears”]]).T #Defining most important features.
tree_model.fit(X=predictors,y=dataset7[“Attrition”]).
with open(“Dtree2.dot”,’w’) as f:
f=tree.export_graphviz(tree_model,feature_names=[“TotatlWorkngYeras”,”Age”,”MonthlyIncome”],out_file=f);
Plot the tree and read it .
Logistic Regression On Bank Loan Dataset:
import statsmodels.api as sm
Y=dataset7.Attrition
X=dataset7[[All impIDV]]
X1=sm.add_constant(x)
Logistic_Attrition=sm.Logit(Y,X1)
result=Logisitc_Attrition.fit()
result.summary()
 

Logistic Regression On Attrition Dataset:
import statsmodels.api as sm
Y=dataset3.PersonalLoan
X=dataset3[[All impIDV]]
X1=sm.add_constant(x)
Logistic_BankLoan=sm.Logit(Y,X1)
result=Logisitc_BankLoan.fit()
result.summary()

# DAY 27 AGENDA

K-Nearest Neighbor
Hands-On In Python of KNN
Support Vector Machine
Hands-on in Python of SVM
Finding Accuracy Score Using  Confusion Matrix in Excel
K-Nearest Neighbor:
It is used to classify the records using Euclidean distance.
It is used to find the closeness of the records.
It is an instance-based learning algorithm.
Based on similar characteristics of objects we can classify the records.
We have 2 records as jay and rina.
We have data of age, income, and no of records.
We will find the Euclidean distance using the formula
That is the square root of the summation of the square of the difference between features of the respondents.
We have data of 6 customers.
DV=Response and IDV= Age, Income<number of credit cards.
We will find the Euclidean distance between Every customer and Dravid.
We will check out of the 5 that are closer to Dravid.
We need to do the iteration many times.
Hands On In Python of KNN:
import pandas
from sklearn import preprocessing 
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import accuracy_score 
from sklear.metrics import confusion_matrix
dataset=pd.read_csv(“Dataset_name.csv”)
le=preprocessing.LabelEncoder()
le.fit(dataset[“sex”])
dataset[“Sex”]=le.transform(dataset[“Sex”]
from sklearn import neighbors 
x=dataset.drop([“Pclass”.”PassengerID”],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
knn=neighnors.KneighborsClassifier(n_neighbors=3) #Chossing the best k is difficult it requires large number of accuracy so we first randomly decide 3 as the Pclass is 3 categories .
Knn.fit(X_train,y_train).score(X_test,y_test)# 46:10 
confusion_matrix(y_test,y_pred) #Gives out the confusion matrix
For Pclass as DV and others as IDV and k 3 we get accuracy score as 85.30%.
For the same as above DV and IDV and k=4, we accuracy score as 83.89%.
For the same as above DV and IDV and k=2, we get the accuracy score as 85.39%.
We can add all the elements of the diagonal of the confusion matrix as they are all correctly Classified records and then add all other elements and find the percentage of the accuracy by dividing the correctly classified dataset by the total number of elements that is adding both the correctly and incorrectly classified records.
Support Vector Machine:
We have 2 categories of SVM that is linear and non-linear SVM.
It is a supervised ML model.
We use it for over-dimensional data.
A hyperplane is a line that divides 2 groups.
When we draw hyperplane we have to see  2 rules.
Rule1: Hyper Plane should divide 2 groups.
Rule2: Whichever point is closer to the hyperplane that point is called support vector.
On the line, the equation is equal to 0.
Below the line, it is less than 0.
Above it is greater than 0.
We can draw n-number of the hyperplane, but we have to choose the correct line that can classify the records properly.
The closest point to the hyperplane is called support vector.
The distance between 2 support vector is called margin.
We can ignore some off the outliers while choosing the correct the hyperplane.
As we can not change the position of the point we can remove that outlier and then choose the hyperplane.
Non-Linear SVM:
In some cases, we have one group covered by another group.
So in this case we go for a multidimensional approach.
We convert the 2D data to multidimensional data.
Kernel trick function is used for conversion from 2D to multidimensional.
Non-Linear is used when we have multi – dimensional data
Hands on in Python of SVM:
import pandas
dataset=pd.read_csv(“SVMTrain.csv”]
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import svm 
le=preprocessing.LableEncoder()
dataset[“Sex”]=le.fit_transform(dataset[“sex”])
y-dataset[“PClass”] #Defining IDV
X=dataset.drop([“Pclass”,”PassengerID”],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)#Splitting training and testing dataset .
clf=svm.SVC(gamma=0.01,C=100) #gamma=0.01 says the record will perform 99.99% accurately and C=100 says it will repeat 100 time.
Clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy.score(y_test,y_pred,normalize=True)
confusion_matrix(y_test,y_pred) #Gives Confusion Matrix 
Finding Accuracy Score Using Confusion Matrix in Excel :
DV=PClass annd remaining Idv we have the accuracy score .
We can use DV=Survived and IDV=Others
We can use DV=Gender and others as IDV
And also use Embarked, Parch and Sib we can separately find the accuracy score.
We can add all the elements of the diagonal of the confusion matrix as they are all correctly Classified records and then add all other elements and find the percentage of the accuracy by dividing the correctly classified dataset by the total number of elements that are adding both the correctly and incorrectly classified records.



# DAY 28 AGENDA

Unsupervised Learning
Association Rule 
Apriori Algorithm
Market Basket Analysis
Hands-On in Python
Unsupervised Learning :
It is a learning algorithm.
Association is a type of Unsupervised learning.
It is used mainly to build a recommendation system.
For example, an online shopping system that suggests more products associated with the current purchase.
Facebook suggests friends associated with your friends.
YouTube also recommends you video associated with the current content you are watching.
It is used to find frequently used combinations.
We have two main things in the Association rule.
First is the support that means the probability of something.
Confidence is the conditional probability of the combination of an itemset.
We have 2 methods that are Aprioiri Algo and Market basket analysis.
Apriori Algo is about frequently occurring itemset.
And MBA is combining fast-moving and slow-moving item set 
Association Rule :
This is used to find Frequently occurring patterns.
Using this we can do cross-marketing, catalog design, clustering.
The item here means a product and transaction mean a set of products.
Transaction Database is a set of transactions.
Support is the probability of buying one object.
Confidence is the conditional probability of buying 2 products together.

Apriori Algorithm:
1. We use this algorithm to find the frequently occurring combination of an itemset .

2.List of products:

Milk 
Jam 
Bread 
Wheat Bread 
Butter  
3. We have 4 transactions into consideration.

4. In the first one, we have milk, bread,wheat bread.

5. In the second one, we have jam, bread, and butter.

6. In the third transaction, we have milk, jam, wheat bread, butter 

7. In the fourth, we have only jam and butter.

8. We will now check-in how many transactions each product was purchased.

9. Wheat bread is the least frequently occurring item so we remove that out.

10. We have 4 products we left on the list.

11. Now we will look for a combination of each item with another.

12. We get 6 different combinations.

13. We will then check the transactions for each combination we have .`

14. In the combination of 2, we will check the least number of occurrences of combination.

15. So we remove 2 combinations.

16. We will make a combination of 3 items and check their occurrence.

17. So we find that jam bread and butter are the most frequently occurring item set.

Market Basket Analysis :
This is when we combine a fast-moving combination with the slow-moving combination.
In the example, we have 400 bread packets we have 200 sweet bread and 200 wheat bread.
In 12 days 180 packets were sold out of sweet bread.
In 12 days 20 packets were sold out of wheat bread.
So we first find the most selling combination with the least selling combination with some off-price we can increase the sale of the least selling object.
We can use an MBA for introducing new products.
WE can attach the product with the old most selling product.
Hands On in Python:
import pandas as pd
dataset=[[“milk”.”break”,”wheat bread”],[“jam”,”bread”,” butter”],.….other transaction]
from mlextend.preprocessing import TransactionEncoder #For converting data to boolean value table
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)
df=pd.DataFrame(te_ary,columns=te.columns_)
print(df) #Gives boolean value table stating which transaction had which items
from mlextend.frequent_patterns import apriori
apriori(df,min_support=0.1)
Gives out how much percentage the combination has occurred that is the support .
frequent_itemsets=apriori(df,min_support=0.1,colnames=True)
frequent_itemsets[‘length’]=frequent_itemsets[‘itemsets’].apply(lambdax:len(x)) #Gives out the length 
frequent_itemsets[(frequent_itemsets[‘length’]==3)&( frequent_itemsets[‘support’]>=0.5)] #Condition to filter out most occurring combination.
Like this, we can frame many other conditions and filter out the data we want to take note of.


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


# Day 34 Agenda :

Natural Language Processing
History of NLP
Need for NLP
Application of NLP
Challenges and Scope in NLP
Natural Language Processing :

It is just the conversion of human language to a language that machine can understand.
It has the most scope in English language .
Sentiment analysis is done using NLP .
It is a branch of AI 
How do Machines and Humans Interact using NLP:

Human talks to machine.
Machine captures audio.
Audio is converted to text .
Processing of the text data is done .
Now the response of the computer is converted to audio.
Machine replies back through audio .
For example Alexa .
History of NLP :

Mr Alan Turing in 1950 made a research paper on the conversion of English to Russian and Russian to English through a machine model.
According to Chomsky the grammar should be in the form A gives BC.
Around 1990 probabilistic and data driven models became quite normal .
The naive bayes model came into picture as the naive bayes is used in NLP .
In 2000 a large quantity of spoken and textual data became accessible .
Human Language :

We need to first understand how properly the human language works.
We need to work on how to make a meaning full sentence using grammar tools and rules.
Need for NLP:

NLP is hallmark of human intelligence .
We have a lot of data being shared and stored every day in the form of text .
All the data on internet is unstructured data .
To make this right we need a huge amount of processing to make the structured data out of unstructured data .
To get all this data we need to use text mining .
Text Mining :

This has 3 parts :
We need to see the structure of the data 
Derive pattern out of the data 
Then evaluate the output 
Application of NLP :

Speech recognition :- Alexa ,Siri.
Machine Translation :- Google Translate .
Chatbot 
Information retrieval 
Information Extraction 
Spell Check 
Question Answering System 
Sentiment Analysis 
Subjective 
Objective 
Challenges and Scope in NLP :

Nature of the human language .
Human language is unstructured data .
Tough to extract meaning from text .
Semantic Meaning :It basically talks of understanding a word with respect to its context .
Entity Extraction :- Extraction of unknown entity .
Anaphora Resolution :Absence of entity in text conversation .
Multiple intents :- User speaking many things in single text .
Word Sense Disambiguation :- Actual context of text .
Notebook of the Project will be shared once it is completed 



# Day 35 Agenda :

Continuation of Sentiment Analysis Hands on .
Deep Learning 
Why Deep learning over Machine Learning 
Perceptron 
Deep Learning :

It is an extended field of ML that has proven useful in the domain of processing text,image,and speech ,primarily .
We have a lot of concept to be learned .
We will be continuously toggling between topics .
The collection of algo implemented under deep learning have similarities with the relationship between stimuli and neurons in the human brain .
We use DL in image recognition/generation ,voice recognition ,text recognition etc.
Why DL Over ML:

ML works fine with less data as the amount of data increases the accuracy increases but after a point it does not increase the performance .
In DL it work not very good with small data but as the size of data increases the performance of the DL model goes on increasing .
So in this era of ever increasing unstructured heaps of data it is important that we data is being utilised for the better good of society .
What is Deep in DL:

The term deep in deep learning refers to the depth of thr artificial neural network architecture,and learning stands for learning through the artificial neural network itself.
Deep network has many layers in the network hence deep as the depth of layer is high and in shallow network there is only 1 layer .
Earlier we used to go to shallow network as it requires less computing power.
If the number of layers go more than 1 it is called as deep network .
What is Deep Neural network Capable of:

Discover latest structures from unlabeled and unstructured data in different forms .
Basic Structure of Neural Network:

Each unit or neuron is simple.
Human brain has 100 billion neurons with 100 trillion connections.
Strength and nature of connection stores memories and program that makes us uman .
A neural network is on web of Artificial Neuron.
We have neurons in human beings and we wanna make a neuron as well for machines .
Artificial neuron or perceptron ,first developed in the 1950s by Frank Rosenblatt.
We will talk about a simple neuron where there are several inputs and those are processed and gives one single output .
There is some kind of processing happening between the input the output .
Perceptron :

This is the first building block of ANN.
We will have number of weights or inputs on the input side .
The body of perceptron does all the working that we have to design for the neuron to work like the Human Neural Network .
In the example if the output is the summation of wixi for all 3 inputs is less than threshold the output will be 0 .
And if it more than threshold the output will be 1 .


# Project Day 1 Agenda :

## Anamoly Detection In Machine Learning

Types of Anomaly Detection
PyOD
Benchmark of Various outlier detection models
Model Building Using
Anomaly Detection In Machine Learning :
So far we have learnt EDA,supervised and unsupervised learning .
WE will get into a real business problem .
We will learn to build real time project .
We will do the first project on Anomaly detection or something Abnormal .
Anomaly is referred to as uncertain behavior .
Objective is identify the anomaly .
In Medical industry Anomaly is a miracle .
In IT industry if there is any attack on the system we will get an abnormal response from our system 
Novelty/Outlier/Forgery and out of distribution detection are all same.
Hawkins defined anomaly as “an abbreviation which deviates sp much from the other observations as to arouse suspicious that it was generated by a different mechanism .
Anomaly detection has received considerable attention in the field of data mining due to the valuable insight that the detection of unusual events can provide in a variety of application 
WE can use this to detect faulty sensor.
In anomaly detection Domain Knowledge is very important .
An example is the breed of dogs if a new species is seen it is novrl class and if any other breed comes in it is outlier .
Types of Anomaly Detection:
Time-series anomaly is like attack on a system .
Video-Level Detection:In Banks,ATM and other important places in cctv recording we can set certain limits and category as to if someone does any unwanted behvior it will set an alarm .
Image-level detection:Can be used in cases where human cannot check the similarity n 2 two images we can find the percentage of similarity in 2 pictures .There are 3 types of categories .
Anomaly Classification target.
Out-of-Distribution Detection.
Anomaly Segmentation Target .
PyOD:
PyOD is a comprehensive and scaleable python toolkit for detecting outlying objects in multivariate data .
It was developed back in 2017 and has been used in many academic research and commercial products .
PyOD Uses:
It is featured for Unified APIs ,detailed documentation and interactive examples across Various Algorithms.
In Advanced models,includinf Neural Networks and outlier Ensembles
Optimized performance with JIT and parallelization when possible,using namba and joblib.
Benchmark of Various outlier detection models:
Linear Models for Outlier Detection :Wehn one increases or decreases with respect to other it is linear.
Principal Component Analysis :Based on the contribution can we remove any of the feature and choose the most important ones is PCA.
Minimum Covariance Determinant :Covariance is the difference bet std deviation and other variance values .So using a limit from the midpoint we can detect outliers after a range.
One-Class Support Vector Machine:WE can take all the inliner and remove the uncertain and out of the line problems.
Proximity Based Outlier Detection Models :Using the proximity to detect the outliers
Local Outlier Factor 
Clustering Based LOF
KNN
Histogram Based Outlier Score
Probability Model for outlier Detection:
Angle-Based Outlier Detection
Ensemble and combination Framework
Isolation Forest
Feature bagging 


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
