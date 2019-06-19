import pandas as pd, numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

dataset = pd.read_csv('kindle_reviews.csv')

dat = dataset.sample(n=20000)

list(dat)
useful = ['helpful','overall', 'reviewText','summary']
dat = dat[useful]
dat[['Thumbsup','ThumbsDown']] = dat['helpful'].str.split(',', expand=True)
dat.Thumbsup = dat.Thumbsup.map(lambda x: str(x)[1:])
dat.ThumbsDown = dat.ThumbsDown.map(lambda x: str(x)[:-1])
dat.drop(columns = ['helpful'],inplace = True)


for i in list(dat):
    print(dat[i].isnull().value_counts())

dat= dat[~(dat['reviewText'].isnull())]
dat= dat[~(dat['summary'].isnull())]

sid = SentimentIntensityAnalyzer()

dat['Sentiment_reviewText'] = dat['reviewText'].map(lambda x: sid.polarity_scores(x))
dat['Negative_reviewText'] = dat['Sentiment_reviewText'].map(lambda x: x['neg'])
dat['Neutral_reviewText'] = dat['Sentiment_reviewText'].map(lambda x: x['neu'])
dat['Positive_reviewText'] = dat['Sentiment_reviewText'].map(lambda x: x['pos'])
dat['compound_reviewText'] = dat['Sentiment_reviewText'].map(lambda x: x['compound'])

#dat.to_csv('Sentiment for review_text.csv')

dat['Sentiment_summary'] = dat['summary'].map(lambda x: sid.polarity_scores(x))
dat['Negative_summary'] = dat['Sentiment_summary'].map(lambda x: x['neg'])
dat['Neutral_summary'] = dat['Sentiment_summary'].map(lambda x: x['neu'])
dat['Positive_summary'] = dat['Sentiment_summary'].map(lambda x: x['pos'])
dat['compound_summary'] = dat['Sentiment_summary'].map(lambda x: x['compound'])

#dat.to_csv('Sentiment for summary_text.csv')

dat[['compound_summary','compound_reviewText']].describe()
list(dat)

Functions
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    num_free = re.sub('^[0-9]+', '', stop_free)
    punc_free = ''.join(ch for ch in num_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


#Positive Word Cloud
pos = list(dat.loc[(dat['compound_reviewText']>0),'summary'])
pos_clean =  [clean(poss) for poss in pos]

text = '.'.join(pos_clean)

wordcloud = WordCloud(width=1366, height=800, colormap="Blues",collocations=False).generate(text)
fig = plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

fig.savefig("Postive_Word_cloud.png")

#Negative Word Cloud
neg = list(dat.loc[(dat['Negative_reviewText']>0.50),'summary'])
neg_clean =  [clean(negs) for negs in neg]

text = '. '.join(neg_clean)

wordcloud = WordCloud(width=1366, height=800, colormap="Oranges",collocations=False).generate(text)
fig2 = plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

fig2.savefig("Negetive_Word_Cloud.png")
#Percentage of Positive and Negative Comments
def tab_sent(dt):
    totp = len(dt[dt>0])
    totn = len(dt[dt<0])
    tot = len(dt)
    print('############### Sentiment Analysis ###############')
    print('No of Positive Comments : ',totp)
    print('No of Negative Comments : ',totn)
    print('Percentage of Positive Comments : ', 100*totp/tot,'%')
    print('Percentage of Negative Comments : ', 100*totn/tot,'%')
    print('##################################################')

tab_sent(dat['compound_summary'])
tab_sent(dat['compound_reviewText'])

#Topic Modelling
######## Topics in Positive Text
##############################Cleaning and Preprocessing#######################################
doc_complete = dat.loc[(dat.compound_reviewText>0),'reviewText']
doc_clean = [clean(doc).split() for doc in doc_complete] 

################################Creating Term Doc Matrix#######################################
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=1)

ldamodel.print_topics(num_topics=3, num_words=10)

topic1_p = ldamodel.print_topics(num_topics=2, num_words=20)[0]
topic1_p = pd.DataFrame({'Topics' : topic1_p[1].split('+')})
topic1_p[['Score','Topic']]  = topic1_p['Topics'].str.split('*',expand=True)
topic1_p['Topic'] = topic1_p['Topic'].str.replace('"','')
topic1_p = topic1_p[['Topic','Score']]

topic2_p = ldamodel.print_topics(num_topics=2, num_words=20)[1]
topic2_p = pd.DataFrame({'Topics' : topic2_p[1].split('+')})
topic2_p[['Score','Topic']]  = topic2_p['Topics'].str.split('*',expand=True)
topic2_p['Topic'] = topic2_p['Topic'].str.replace('"','')
topic2_p = topic2_p[['Topic','Score']]

topic3_p = ldamodel.print_topics(num_topics=3, num_words=20)[1]
topic3_p = pd.DataFrame({'Topics' : topic3_p[1].split('+')})
topic3_p[['Score','Topic']]  = topic3_p['Topics'].str.split('*',expand=True)
topic3_p['Topic'] = topic3_p['Topic'].str.replace('"','')
topic3_p = topic3_p[['Topic','Score']]

######## Topics in Negative Text
##############################Cleaning and Preprocessing#######################################
doc_complete = dat.loc[(dat.compound_reviewText<0),'reviewText']
doc_clean = [clean(doc).split() for doc in doc_complete] 

################################Creating Term Doc Matrix#######################################
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=1)

ldamodel.print_topics(num_topics=3, num_words=10)

topic1 = ldamodel.print_topics(num_topics=2, num_words=20)[0]
topic1 = pd.DataFrame({'Topics' : topic1[1].split('+')})
topic1[['Score','Topic']]  = topic1['Topics'].str.split('*',expand=True)
topic1['Topic'] = topic1['Topic'].str.replace('"','')
topic1 = topic1[['Topic','Score']]

topic2 = ldamodel.print_topics(num_topics=2, num_words=20)[1]
topic2 = pd.DataFrame({'Topics' : topic2[1].split('+')})
topic2[['Score','Topic']]  = topic2['Topics'].str.split('*',expand=True)
topic2['Topic'] = topic2['Topic'].str.replace('"','')
topic2 = topic2[['Topic','Score']]

topic3 = ldamodel.print_topics(num_topics=3, num_words=20)[1]
topic3 = pd.DataFrame({'Topics' : topic3[1].split('+')})
topic3[['Score','Topic']]  = topic3['Topics'].str.split('*',expand=True)
topic3['Topic'] = topic3['Topic'].str.replace('"','')
topic3 = topic3[['Topic','Score']]