import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('15/udemy_courses_data.csv')

plt.figure(figsize=(15,7))
sn.countplot(df['subject'],palette='plasma')

plt.figure(figsize=(10,5))

df['subject'].value_counts().plot(kind='pie')

df.groupby('subject')['num_subscribers'].count().plot(kind = 'barh')
df.groupby('subject')['num_subscribers'].sum().plot(kind = 'barh')
df.groupby('subject')['num_subscribers'].sum().plot(kind = 'pie')

df['level'].value_counts().plot(kind = 'barh')
df.groupby(['level'])['num_subscribers'].count().plot(kind='barh')
df.groupby(['level'])['num_subscribers'].sum().plot(kind = 'bar')
df.groupby(['subject'])['level'].count().plot(kind = 'bar',color = 'red')

plt.figure(figsize=(15,7))
df.groupby(['subject'])['level'].value_counts().plot(kind = 'bar')

plt.xticks(fontsize = 20,fontweight = 'bold')
plt.yticks(fontsize = 20,fontweight = 'bold')
plt.xlabel('Subject Category',fontsize = 20,fontweight = 'bold',color = 'blue')
plt.ylabel('Count of Levels',fontsize = 20,fontweight = 'bold',color = 'blue')

plt.figure(figsize=(16,9))
sn.barplot(x='level',y='num_subscribers', hue='subject',data=df)
plt.xticks(fontsize = 20,fontweight = 'bold')
plt.yticks(fontsize = 20,fontweight = 'bold')
plt.xlabel('Level',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Subscribers Count',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.xticks(rotation = 'vertical')

plt.figure(figsize=(16,9))
sn.barplot(x='level',y='num_lectures', hue='subject',data=df)
plt.xticks(fontsize = 20,fontweight = 'bold')
plt.yticks(fontsize = 20,fontweight = 'bold')
plt.xlabel('Level',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Lectures Count',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.xticks(rotation = 'vertical')

def plotdata(df,feature):
    plt.figure(figsize=(10,7))
    plt.title("Plot of {} per level per subject".format(feature))
    sn.barplot(x = 'level',y = feature,data=df,hue = 'subject')
    plt.show()
    
    
featureslist = ['num_subscribers', 'num_reviews', 'num_lectures']
for feature in featureslist:
    plotdata(df,feature)

df['price'] = df['price'].str.replace('TRUE|Free','0')
df['price'] = df['price'].astype('float')

plt.figure(figsize=(15,7))
df['price'].value_counts().plot(kind = 'bar',color = 'violet')

df['profit'] = df['price'] * df['num_subscribers']

df[df['profit']==df['profit'].max()].style.background_gradient(cmap = 'plasma')
df[df['profit']==df['profit'].min()].style.background_gradient(cmap = 'plasma')
df[df['profit']>=df['profit'].mean()].style.background_gradient(cmap = 'plasma')

paid_dict = {'True':True,'False':False,'FALSE':False,
             'TRUE':True,'https://www.udemy.com/learnguitartoworship/':True}

df['is_paid'] = df['is_paid'].map(paid_dict)

sn.countplot(x = 'is_paid',data=df,palette='plasma')

df.groupby(['is_paid'])['subject'].value_counts()

df[df['is_paid']==True].groupby(['is_paid'])['subject'].value_counts().plot(kind = 'bar',color = 'magenta')

df[df['is_paid']==False].groupby(['is_paid'])['subject'].value_counts().plot(kind = 'bar',color = 'magenta')
sn.scatterplot(data=df,x='price',y='num_subscribers')
sn.scatterplot(data=df,x = 'price',y = 'num_reviews')

plt.figure(figsize=(20,10))
plt.title("Does Price Influence Subscription Per Subject Category",
         fontsize = 20,fontweight = 'bold')

sn.lineplot(data=df,x='price',y='num_subscribers',hue='subject')

plt.xticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.xlabel('Price',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Number of Subscribers',fontsize = 20,fontweight = 'bold',
           color = 'blue')

df['published_date'] = df['published_timestamp'].apply(lambda x:x.split('T')[0])
df['published_date'] = pd.to_datetime(df['published_date'],format="%Y-%m-%d")

df[df['published_date']=='3 hours']
df = df.drop(df.index[2066])


df['published_date'] = pd.to_datetime(df['published_date'],format="%Y-%m-%d")
df['Year'] = df['published_date'].dt.year

df['Month'] = df['published_date'].dt.month

df['Day'] = df['published_date'].dt.day

df['Month_name'] = df['published_date'].dt.month_name()

plt.figure(figsize=(15,7))
df.groupby(['Year'])['profit'].sum().plot(kind = 'bar')
plt.xticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.xlabel('Year',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Profit',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.figure(figsize=(15,7))
df.groupby(['Month_name'])['profit'].sum().plot(kind = 'bar')
plt.xticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.xlabel('Year',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Profit',fontsize = 20,fontweight = 'bold',
           color = 'blue')

plt.figure(figsize=(15,7))
df.groupby(['Year'])['num_subscribers'].sum().plot(kind = 'bar',color = 'green')

plt.xticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.xlabel('Year',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Number of Subscribers',fontsize = 20,fontweight = 'bold',
           color = 'blue')

plt.figure(figsize=(15,7))
df.groupby(['Month_name'])['num_subscribers'].sum().plot(kind = 'bar',
                                                         color = '#b57edc')

plt.xticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 20,fontweight = 'bold',rotation = 45)
plt.xlabel('Year',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Number of Subscribers',fontsize = 20,fontweight = 'bold',
           color = 'blue')

df[df['course_title'].str.len()==max(df['course_title'].str.len())]
df['course_title'].iloc[2190]
df[df['num_reviews']==max(df['num_reviews'])]
df[df['course_title'].str.len()==min(df['course_title'].str.len())]
df['course_title'].iloc[1327]

import neattext.functions as nfx

df['Clean_title'] = df['course_title'].apply(nfx.remove_shortwords)
df['Clean_title'].iloc[1:5]

temp = df[['Clean_title','course_title']]

temp[temp['Clean_title'].str.len()==max(temp['Clean_title'].absstr.len())]
temp['Clean_title'].iloc[293]
temp['course_title'].iloc[293]

df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)
df['Clean_title'].iloc[1:5]

all_title_list = df['Clean_title'].to_list()

all_title_list[1:5]

alltokens = [token for line in all_title_list for token in line.split()]

print(len(alltokens))

print(alltokens[1:4])

from collections import Counter

freq_words = dict(Counter(alltokens).most_common(50))

freq_words

plt.figure(figsize=(30,15))
plt.bar(*zip(*freq_words.items()))
plt.xticks(rotation = 45)

plt.xticks(fontsize = 15,fontweight = 'bold',rotation = 45)
plt.yticks(fontsize = 10,fontweight = 'bold',rotation = 45)
plt.xlabel('Frequent Words',fontsize = 20,fontweight = 'bold',
           color = 'blue')
plt.ylabel('Count of Frequent Words',fontsize = 20,fontweight = 'bold',
           color = 'blue')

plt.show()

from rake_nltk import Rake
rake = Rake()

allwords = ''.join(alltokens)

rake.extract_keywords_from_text(allwords)

rake.get_ranked_phrases_with_scores()