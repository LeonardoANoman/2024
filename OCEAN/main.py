import numpy as np
import pandas as pd
import gradio as gr

from matplotlib import pyplot as plt
from io import open
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

pd.options.display.max_columns = 150
pd.options.display.float_format = "{:.2f}".format

# You may download the files from
# https://www.kaggle.com/datasets/tunguz/big-five-personality-test

data = pd.read_csv("./Kaggle/data-final.csv", sep='\t')

data.head()

data.drop(data.columns[50:110], axis=1, inplace=True)

data.describe()

data[(data == 0.00).all(axis=1)].describe()

data = data[(data > 0.00).all(axis=1)]

kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,10))

data_sample = data.sample(n=5000, random_state=1)

visualizer.fit(data_sample)

kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data)

predictions = k_fit.labels_
data['Clusters'] = predictions

col_list = list(data)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

data_sum = pd.DataFrame()
data_sum['extroversion'] = data[ext].sum(axis=1)/10
data_sum['neurotic'] = data[est].sum(axis=1)/10
data_sum['agreeable'] = data[agr].sum(axis=1)/10
data_sum['conscientious'] = data[csn].sum(axis=1)/10
data_sum['open'] = data[opn].sum(axis=1)/10
data_sum['clusters'] = predictions

data_clusters = data_sum.groupby('clusters').mean()

plt.figure(figsize=(22,3))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color='green', alpha=0.2)
    plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color='red')
    plt.title('Group ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4)

question_list = open("./Kaggle/codebook.txt").read().split("\n")

questions = []
for q in question_list:
    q = str(q)
    questions.append(q[q.find("\t"):].lstrip())

label_dict = {
    0: 'extroversion',
    1: 'neurotic',
    2: 'agreeable',
    3: 'conscientious',
    4: 'open'
}

input_questions = []
for q in questions:
    obj_input = gr.components.Slider(minimum=1, maximum=5, step=1, value=3, label=q)
    input_questions.append(obj_input)

def predict(*output_questions):
    output_questions = np.array(output_questions).reshape(1, -1)
    cluster_label = k_fit.predict(output_questions)[0]
    predicted_trait = label_dict[cluster_label]
    return predicted_trait

iface = gr.Interface(
    fn=predict,
    title="Big Five Personality Test",
    description="System for detecting personality traits.",
    inputs=input_questions,
    outputs=gr.components.Textbox()
)

iface.launch(share=True)
