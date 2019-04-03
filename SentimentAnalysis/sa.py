import numpy as np
import pandas as pd
import textblob 
from textblob import TextBlob
import warnings
from nltk.corpus import twitter_samples
from textblob.classifiers import NaiveBayesClassifier
warnings.filterwarnings("ignore")


#SOURCE: http://blog.chapagain.com.np/python-twitter-sentiment-analysis-using-textblob/


tweets = 'trump.csv'


df = pd.read_csv(tweets, error_bad_lines=False, sep= ';', encoding='cp1252' )

#df = df.dropna()
#print(df.head(10))

df['text'] = df['text'].astype('str')
print (twitter_samples.fileids())
pos_tweets = twitter_samples.strings('positive_tweets.json')
pos_tweets = twitter_samples.strings('positive_tweets.json')
print (len(pos_tweets)) # Output: 5000
 
neg_tweets = twitter_samples.strings('negative_tweets.json')
print (len(neg_tweets)) # Output: 5000
 
#all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
#print (len(all_tweets)) # Output: 20000
 
# positive tweets words list
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((tweet, 'pos'))
 
# negative tweets words list
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((tweet, 'neg'))
 
print (len(pos_tweets_set), len(neg_tweets_set)) # Output: (5000, 5000)
print(pos_tweets_set[:5])




list_tweets = []
for i in range(0, df['text'].count()):
    str1 = df['text'][i]
    list_tweets.append(str1)
#print(list_tweets[10])

train_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
test_set =  pos_tweets_set[1000:5000] + neg_tweets_set[1000:5000]

 
print(len(test_set),  len(train_set))
classifier = NaiveBayesClassifier(train_set)
print("All CLEAR")


list_sentiment = []
for i in range(0, df['text'].count()):
    sentiment1 = TextBlob(list_tweets[i], classifier=classifier)
    list_sentiment.append(sentiment1.classify())

print(list_sentiment[df['text'].count() - 1])
df['sentiment'] = list_sentiment



#default polarity classifier of TextBlob
list_polarity = []
for i in range(0, df['text'].count()):
	polarity1 = TextBlob(list_tweets[i])
	list_polarity.append(polarity1.sentiment.polarity)

df['polarity_score'] = list_polarity

list_subjectivity = []
for i in range(0, df['text'].count()):
	subjectivity1 = TextBlob(list_tweets[i])
	list_subjectivity.append(subjectivity1.sentiment.subjectivity)

df['subjectivity_score'] = list_subjectivity

print(df.head())

df.to_csv('trump_sa.csv', sep= ';', encoding='utf-8')
