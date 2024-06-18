import pandas as pd 
import numpy as np 
import seaborn as sn 
import matplotlib.pyplot as plt 
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle

df = pd.read_csv("17/bodyfat.csv")

def plotdistplots(col):
    plt.figure(figsize=(12,5))
    sn.distplot(df['BodyFat'], color='magenta', hist=False, label='BodyFat')
    sn.distplot(df[col], color='red',hist=False,label=col)
    plt.legend()
    plt.show()

cols = list(df.columns)

"""
for i in cols:
    print(f'Distribution plots for {i} feature i shown')
    plotdistplots(i)
    print('-'*100)

"""

def drawplots(df,col):
    plt.figure(figsize=(15,7))
    plt.subplot(1,3,1)
    plt.hist(df[col], color='magenta')

    plt.subplot(1,3,2)
    stats.probplot(df[col], dist='norm', plot=plt)

    plt.subplot(1,3,3)
    sn.boxplot(df[col], color='magenta')

    plt.show()

"""
for i in range(len(cols)):
    print(f"Distribution plots for the feature {cols[i]} are shown")
    drawplots(df,cols[i])
    print('='*100)

"""

upperlimit = []
lowerlimit = []
for i in df.columns:
    upperlimit.append(df[i].mean()+(df[i].std())*4)    
    lowerlimit.append(df[i].mean()-(df[i].std())*4)    
    
j = 0
for i in range(len(cols)):
    temp = df.loc[(df[cols[i]]>upperlimit[j])&(df[cols[i]]<lowerlimit[j])]
    j += 1
    # print(temp)

data = df.copy()
test = data['BodyFat']
train = data.drop(['BodyFat'], axis=1)

er = ExtraTreesRegressor()
er.fit(train, test)

series = pd.Series(er.feature_importances_, index=train.columns)
series.nlargest(5).plot(kind= 'barh', color= 'green')

mr = mutual_info_regression(train, test)
plotdata = pd.Series(mr, index=train.columns)
plotdata.nlargest(5).plot(kind= 'barh', color='green')

def correlation(df,threshold):
    colcor = set()
    cormat = df.corr()

    for i in range(len(cormat)):
        for j in range(i):
            if abs(cormat.iloc[i][j])>threshold:
                colname = cormat.columns[i]
                colcor.add(colname)

    return colcor

ans = correlation(train,threshold=0.85)

temp = data[list[data.columns]]
info = pd.DataFrame()
info["VIF"] = [variance_inflation_factor(temp.values, i) for i in range(temp.shape[1])]
info['Column'] = temp.columns

cols1 = list(series.nlargest(5).index)
cols2 = list(plotdata.nlargest(5).index)

totrain = train[cols1]

X_train, X_test, y_train, y_test = train_test_split(totrain, test, test_size=0.2)

reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

path = reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alpha = path.ccp_alphas

alphalist = []
for i in range(len(ccp_alpha)):
    reg = DecisionTreeRegressor(ccp_alpha=ccp_alpha[i])
    reg.fit(X_train, y_train)
    alphalist.append(reg)

trainscore = [alphalist[i].score(X_train, y_train) for i in range(len(alphalist))]
testscore = [alphalist[i].score(X_test, y_test) for i in range(len(alphalist))]

clf = DecisionTreeRegressor(ccp_alpha=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f'Decision Tree Normal Approach: {metrics.r2_score(y_test, y_pred)}')

rf = RandomForestRegressor(n_estimators=1000,ccp_alpha=1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'Random Forest Normal Approach:  {metrics.r2_score(y_test, y_pred_rf)}')

params = {
    'RandomForest':{
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [int(x) for x in np.linspace(start=1, stop=1200, num=10)],
            'criterion': ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'],
            'max_depth': [int(x) for x in np.linspace (start=1, stop=30, num=5)],
            'min_samples_split': [2,5,10,12],
            'min_samples_leaf': [2,5,10,12],
            'max_features': ['auto', 'sqrt'],
            'ccp_alpha':[1,2,2.5,3,3.5,4,5],
        }
},

    'D-tree':{
        'model' : DecisionTreeRegressor(), 
        'params': {
            'criterion': ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'], 
            'splitter': ['best', 'random'], 
            'min_samples_split':[1,2,5,10,12], 
            'min_samples_leaf':[1,2,5,10,12],
            'max_features': ['auto', 'sqrt'],
            'ccp_alpha':[1,2,2.5,3,3.5,4,5],
    }
},

'SVM':{
    'model': SVR(),
    'params': {
        'C': [0.25,0.50,0.75, 1.0],
        'tol': [1e-10, 1e-5,1e-4,0.025,0.50,0.75],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'max_iter': [int(x) for x in np.linspace (start=1, stop=250, num=10)],
        }
    }
}

scores = []
for modelname, mp in params.items():
    clf= RandomizedSearchCV(mp['model'], param_distributions=mp['params'],
                            cv=5, n_jobs=-1, n_iter=10, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    scores.append({
                   'model_name': modelname,  
                   'best_score': clf.best_score_,
                   'best_estimator': clf.best_estimator_
    })


scoresdf = pd.DataFrame(scores, columns=['model_name', 'best_score', 'best_estimator'])
rf = RandomForestRegressor(n_estimators=1000,ccp_alpha=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(metrics.r2_score(y_test, y_pred))
print(totrain)

totrainlist = np.array(totrain)
predicted = []
for i in range(len(totrainlist)):
    predicted.append(rf.predict(totrainlist[i].reshape(1,-1)))

totrain['Actual Result'] = test
totrain['Predicted Result'] = np.array(predicted)
file = open('17/bodyfatmodel.pkl', 'wb')
pickle.dump(rf, file)
file.close()