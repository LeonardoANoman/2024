import pandas as pd


def getvaluecounts(df):
    return dict(df['subject'].value_counts())


def getlevelcount(df):
    return dict(list(df.groupby(['level'])['num_subscribers'].count().items())[1:])


def getsubjectsperlevel(df):
    ans = list(dict(df.groupby(['subject'])['level'].value_counts()).keys())
    alllabels = [ans[i][0]+'_'+ans[i][1] for i in range(len(ans))]
    ansvalues = list(dict(df.groupby(['subject'])[
                     'level'].value_counts()).values())

    completedict = dict(zip(alllabels, ansvalues))

    return completedict


def yearwiseprofit(df):
    for idx, obj in df.iterrows():
        df['price'] = obj['price'].replace('TRUE|Free', '0')
    df['price'] = df['price'].astype('float')
    df['profit'] = df['price'] * df['num_subscribers']

    df['published_date'] = df['published_timestamp'].apply(
        lambda x: x.split('T')[0])

    df = df.drop(df.index[2066])

    df['published_date'] = pd.to_datetime(
        df['published_date'], format="%Y-%m-%d")

    df['Year'] = df['published_date'].dt.year

    df['Month'] = df['published_date'].dt.month

    df['Day'] = df['published_date'].dt.day

    df['Month_name'] = df['published_date'].dt.month_name()

    profitmap = dict(df.groupby(['Year'])['profit'].sum())

    subscribersmap = dict(df.groupby(['Year'])['num_subscribers'].sum())

    profitmonthwise = dict(df.groupby(['Month_name'])['profit'].sum())

    monthwisesub = dict(df.groupby(['Month_name'])['num_subscribers'].sum())

    return profitmap, subscribersmap,profitmonthwise,monthwisesub
