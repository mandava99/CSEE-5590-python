from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
#Reading 'iris' dataset
data_set=pd.read_csv("iris.csv")
#Preprocessing
x=data_set.drop('class',axis=1)
y=data_set['class']
#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
#Training the Gaussian Naive bayes classifier
model=GaussianNB()
model.fit(x_train,y_train)
#Making Predictions
y_prediction1=model.predict(x_test)
clf=SVC(kernel='linear',C=1).fit(x_train,y_train)
#Making predictions
y_prediction2=clf.predict(x_test)
#Evaluating results fo
print(classification_report(y_test,y_prediction1))
print("Gaussian Naive Bayes model accuracy is: ",metrics.accuracy_score(y_test,y_prediction1)*100,"%")

print(classification_report(y_test,y_prediction2))
print("Linear SVM classifier accuracy is: ",metrics.accuracy_score(y_test,y_prediction2)*100,"%")