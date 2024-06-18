import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("Social_Network_Ads.csv")

df.head()
df.isnull().sum()
df.drop(['User ID'],axis=1,inplace=True)

df['Gender'].value_counts()

fig, axes = plt.subplots(1,2,figsize=(12,5))

sns.countplot(x='Gender', data=df, ax=axes[0])
axes[0].set_title('Gender Distribution in dataset')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')

sns.countplot(data=df, x='Purchased', hue='Gender',ax=axes[1], palette='Set2')
axes[1].set_xlabel('Purchased')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Gender by Purchase')

fig, ax = plt.subplots(figsize=(7,5))

sns.histplot(data=df, x="Age", bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Age")

fig, axes = plt.subplots(1,2,figsize=(12,5))

sns.boxplot(x='Purchased', y='Age', data=df, palette='Set2',ax=axes[0])
axes[0].set_xlabel('Purchased')
axes[0].set_ylabel('Age')
axes[0].set_title('Distribution of Age by Purchase')

sns.boxplot(x='Gender', y='Age', data=df,ax=axes[1])
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Age')
axes[1].set_title('Distribution of Age by Gender')


fig, ax = plt.subplots(figsize=(7, 5))

sns.histplot(data=df, x="EstimatedSalary", bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Estimated Salary")


fig, axes = plt.subplots(1,2,figsize=(12,5))

sns.boxplot(x='Gender', y='EstimatedSalary', data=df, palette='Set1', ax=axes[0])
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Estimated Salary')
axes[0].set_title('Distribution of Estimated Salary by Gender')


sns.boxplot(x='Purchased', y='EstimatedSalary', data=df, palette='Set2', ax=axes[1])
axes[1].set_xlabel('Purchased')
axes[1].set_ylabel('EstimatedSalary')
axes[1].set_title('Distribution of Estimated Salary by Purchased')

plt.figure(figsize=(7,5))
not_purchased = df[df['Purchased']==0]
purchased = df[df['Purchased']==1]
plt.scatter(purchased['Age'], purchased['EstimatedSalary'], c='blue', marker='*', label='Purchased')
plt.scatter(not_purchased['Age'], not_purchased['EstimatedSalary'], c='red', marker='o', label='Not Purchased')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Correlation between Age and Estimated Salary')
plt.legend()

ohe = OneHotEncoder(sparse_output=False, drop='first')
male = pd.DataFrame(ohe.fit_transform(df[['Gender']]), columns=['Male'])
male.head()

df['Gender_Male'] = male
df.drop('Gender',axis=1,inplace=True)
df.head()

corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Heatmap')


X = df.drop('Purchased',axis=1)
X.head()

y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled[0:5]

cls = SVC()
cls.fit(X_train_scaled,y_train)

X_test_scaled = scaler.transform(X_test)
y_pred = cls.predict(X_test_scaled)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')

print(classification_report(y_test,y_pred))

X = df.drop(['Gender_Male','Purchased'],axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[0:5]

cls1 = SVC()
cls1.fit(X_train_scaled,y_train)

y_pred = cls1.predict(X_test_scaled)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')


print(classification_report(y_test,y_pred))
    
test = np.array([[27, 150]])
test_scaled = scaler.transform(test)
print(cls1.predict(test_scaled))

test1 = np.array([[27, 350000]])
test_scaled1 = scaler.transform(test1)
print(cls1.predict(test_scaled1))