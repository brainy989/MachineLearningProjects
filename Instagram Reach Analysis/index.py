import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("Instagram.csv", encoding='latin1')

# plt.figure(figsize=(10, 8))
# plt.style.use('fivethirtyeight')
# plt.title("Distribution of Impressions From Home")
# sns.histplot(data['From Hashtags'])
# plt.show()

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

# labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
# values = [home, hashtags, explore, other]

# fig = px.pie(data, values=values, names=labels,
#              title='Impressions on Instagram Posts From Various Sources', hole=0.5)
# fig.show()

# text = " ".join(i for i in data.Caption)
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# plt.style.use('classic')
# plt.figure(figsize=(12, 10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# figure = px.scatter(data_frame=data, x="Impressions",
#                     y="Likes", size="Likes", trendline="ols",
#                     title="Relationship Between Likes and Impressions")

# figure.show()

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)