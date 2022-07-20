#from urllib import request
#url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv'
#request.urlretrieve(url,'cust-churn.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

df=pd.read_csv('cust-churn.csv')
df.columns
df.head()

#using some specific columns
churn_df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head

#cleaning the data
#converting the churn column as integer because that's what we want.
churn_df['churn']=churn_df['churn'].astype('int')

#x=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']].values
#x
x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
x

#y=churn_df[['churn']].values
#y 
y = np.asanyarray(churn_df['churn'])
y


from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]

#train test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)


lr=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)

#will go into an error because y is not flattened, so use: 


#y=y.flatten()
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=40)

#lr=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
lr
y_pred=lr.predict(x_test)
y_pred

#probability of class 0 and class 1
y_prob=lr.predict_proba(x_test)

#accuracy

print(jaccard_score(y_test, y_pred,pos_label=0))

#r2_score does not work for logistic regression


cm= confusion_matrix(y_test, y_pred)
print('confusion matrix=',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.xlabel('Predicted churn')
plt.ylabel('Real churn')
plt.text(-0.3,-0.3,'true positives')
plt.text(0.7,-0.3,'false positives')
plt.text(-0.3,0.7,'false negatives')
plt.text(0.7,0.7,'true negatives')

plt.savefig('confusion-matrix.png',dpi=300)
#plt.show()

print ('classification report=',classification_report(y_test, y_pred))

from sklearn.metrics import log_loss

print('log loss=',log_loss(y_test, y_prob))





