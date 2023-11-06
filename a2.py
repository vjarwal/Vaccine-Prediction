import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings("ignore")



data=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")

print(data.sample(10))
print("\n")
print(data.dtypes)
print("\n")
print(data.shape)
print("\n")
print(data.describe)
print('\n')
print(data.info())

print(data['income_level'].value_counts())
print(data['no_of_adults'].value_counts())
print(data['no_of_children'].value_counts())
print(data['h1n1_worry'].value_counts())
print(data['h1n1_awareness'].value_counts())

X=data.drop(['unique_id','income_level'],axis=1)
y=data['unique_id']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1,random_state=0)


plt.figure(figsize=(60,20))
sns.heatmap(data.isnull(),yticklabels=False)
plt.show()


sns.countplot(x="no_of_adults",hue="income_level",data=data)
plt.show()

sns.countplot(x="no_of_children",hue="income_level",data=data)
plt.show()

sns.countplot(x="h1n1_worry",hue="income_level",data=data)
plt.show()

sns.countplot(x="h1n1_awareness",hue="income_level",data=data)
plt.show()

abc1=AdaBoostClassifier(random_state=0)
print(abc1.score)

gdc1=GradientBoostingClassifier(random_state=0)
print(gdc1.score)

model=LogisticRegression(solver='liblinear')
print(model.score)

dtree=DecisionTreeClassifier(criterion='gini')
print(dtree.score)


rds=RandomForestClassifier(criterion='gini')
print(rds.score)

bg=BaggingClassifier()
print(bg.score)

