import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

employee_df = pd.read_csv('01/Human_Resources.csv')

employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

employee_df_numeric = employee_df.select_dtypes(include=[np.number])

sns.heatmap(employee_df_numeric.isnull(), yticklabels=False, cbar=False, cmap='Blues')

employee_df_numeric.hist(bins=30, figsize=(20, 20), color='red')

employee_df_numeric.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

left_df = employee_df_numeric[employee_df_numeric['Attrition'] == 1]
stayed_df = employee_df_numeric[employee_df_numeric['Attrition'] == 0]

print('Number of employees who left = ', len(left_df))
print('% of employees who left = ', 1. * len(left_df) / len(employee_df_numeric) * 100, '%')

print('Number of employees who stayed = ', len(stayed_df))
print('% of employees who stayed = ', 1. * len(stayed_df) / len(employee_df_numeric) * 100, '%')

correlations = employee_df_numeric.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)

plt.figure(figsize= [25,12])
sns.countplot(x = 'Age', hue = 'Attrition', data= employee_df)

plt.figure(figsize= [20,20])

plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data= employee_df)

plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data= employee_df)

plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data= employee_df)

plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data= employee_df)

plt.figure(figsize=(12,7))
sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who stayed', shade = True, color = 'b')

plt.figure(figsize=(12,7))
sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who stayed', shade = True, color = 'b')

plt.figure(figsize=(12,7))
sns.kdeplot(left_df['TotalWorkingYears'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], label = 'Employees who stayed', shade = True, color = 'b')

sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data=employee_df)

plt.figure(figsize=(15,10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data=employee_df)

X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
onehotenconder = OneHotEncoder()
X_cat = onehotenconder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 
                           'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	
                           'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	
                           'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	
                           ,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]

X_cat.columns = X_cat.columns.astype(str)  
X_numerical.columns = X_numerical.columns.astype(str) 

X_all = pd.concat([X_cat, X_numerical], axis = 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)
y = employee_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Acurracy {}%'.format(100 * accuracy_score(y_pred, y_test)))

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot= True)

print("Logistic Regression")
print(classification_report(y_test, y_pred))

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest")
print(classification_report(y_test, y_pred))

model = Sequential()
model.add(Dense(units=500, activation='relu', input_shape=(50, )))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))