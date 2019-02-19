from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
#Reading the 'iris' dataset
data_set=pd.read_csv("iris.csv")
#Preprocessing
x=data_set.drop('class',axis=1)
y=data_set['class']
#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
#Training the classifier
clf=SVC(kernel='linear',C=1).fit(x_train,y_train)
#Making Predictions
y_prediction=clf.predict(x_test)
#Evaluating the model
print(accuracy_score(y_test,y_prediction))
print(confusion_matrix(y_test,y_prediction))