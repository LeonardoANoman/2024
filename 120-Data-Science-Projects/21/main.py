import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

"""
DATA
https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data

"""


names = ['Class', 'id', 'Sequence']
data = pd.read_csv('../21/promoters.csv', names = names)
classes = data.loc[:,'Class']
sequence = list(data.loc[:, 'Sequence'])

dic = {}
for i, seq in enumerate(sequence):
    nucleotides = list(seq)
    nucleotides = [char for char in nucleotides if char != '\t']
    nucleotides.append(classes[i])
    
    dic[i] = nucleotides

df = pd.DataFrame(dic)
df = df.transpose()
df.rename(columns = {57:'Class'}, inplace = True)
temp = df.copy(deep=True)
temp = temp.drop(['Class'], axis = 1)

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(temp)
df1 = enc.transform(temp).toarray()
del temp

file = open('../21/EColi-encoder.pickle','wb')
pickle.dump(enc,file)
file.close()

df_new = pd.DataFrame(df1)

df["Class"] = df["Class"].replace(to_replace =["+"], value =1)
df["Class"] = df["Class"].replace(to_replace =["-"], value =0)
df_new["Classes"] = df['Class']

numerical_df = pd.get_dummies(df)

y = df_new['Classes'].values
X = df_new.drop(['Classes'], axis = 1).values

seed = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = seed)

model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

model.fit(X_train, y_train)
print(model.score(X_train, y_train))

y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

print(classification_report(y_test, y_pred))

loss_values = model.loss_curve_
plt.plot(loss_values)
plt.show()

file = open('../21/EColi-model.pickle','wb')
pickle.dump(enc,file)
file.close()

genome = "ttactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcg"
genome_list = list(genome)
df_test = pd.DataFrame(genome_list)
df_test = df_test.transpose()

encoder = pickle.load(open("../21/EColi-encoder.pickle", 'rb')) 
data_test = encoder.transform(df_test).toarray()
print(model.predict(data_test))