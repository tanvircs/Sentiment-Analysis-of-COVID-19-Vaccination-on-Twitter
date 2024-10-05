#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re    # for regular expressions  
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

get_ipython().magic('matplotlib inline')


# In[2]:


import json
import csv
import nltk
import tweepy
from sklearn import metrics
import re
import os
from nltk.corpus import stopwords
from keras.layers import Dense, Dropout, Activation
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim
from tqdm import tqdm 
tqdm.pandas(desc="progress-bar") 
from gensim.models.doc2vec import TaggedDocument


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout , Activation, Flatten, SimpleRNN
from keras.layers import GlobalMaxPool1D
from keras.models import Model, Sequential
import tensorflow as tf

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[4]:


df1 = pd.read_csv('final_data.csv')


# In[5]:


df1.head(0)


# In[6]:


df1['split'] = np.random.randn(df1.shape[0], 1)

msk = np.random.rand(len(df1)) <= 0.7

train_1 = df1[msk]
test_1 = df1[~msk]


# In[7]:


train_1.to_csv('final_train.csv', encoding='utf-8')


# In[8]:


test_1.to_csv('final_test.csv', encoding='utf-8')


# In[9]:


train  = pd.read_csv('final_train.csv') 
test = pd.read_csv('final_test.csv')


# In[10]:


train.drop(train.columns[[0,1]], axis=1, inplace=True)


# In[11]:


test.drop(test.columns[[0,1]], axis=1, inplace=True)


# ## Exploring Data

# In[12]:


train.info()


# In[13]:


test.head()


# In[14]:


train.shape, test.shape


# In[15]:


train["sentiment"].value_counts()


# In[16]:


plt.hist(train.text.str.len(), bins=20, label='train')
plt.hist(test.text.str.len(), bins=20, label='test')
plt.legend()
plt.title('Tarin Test distribution in dataset', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.tight_layout()
plt.tight_layout()
plt.savefig("Train Test Distribution.png", dpi=200, bbox_inches='tight')


# In[17]:


combi = train.append(test, ignore_index=True, sort=True)
combi.shape


# In[18]:


combi['sentiment'].replace({1: 2, 0: 1, -1: 0}, inplace=True)


# In[19]:


combi["sentiment"].value_counts()


# In[20]:


combi.drop(combi.columns[[6]], axis=1, inplace=True)


# In[21]:


print("Number of rows per vaccine rating:")
print(combi['sentiment'].value_counts())
plt.figure(figsize=(10, 7)) 
combi['sentiment'].value_counts().sort_index().plot.bar()
plt.title('Sentiment distribution in df', fontsize=20)
plt.xlabel('Sentiment Label', fontsize=18)
plt.ylabel('Tweet Count', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig("Sentiment_Distribution.png", dpi=200, bbox_inches='tight')


# ## Data Wrangling & Preprocessing
# ### Remove Punctuation: (a)hashtags (b)links (c)punctuations (d)non-alphanumeric characters

# In[22]:


def preprocess_word(word):
    word = word.lower()
    word = re.sub("'", "", word)
    word = word.strip('\'"?!,.():;')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub("@[A-Za-z0-9_]+","", word)
    word = re.sub("#[A-Za-z0-9_]+","", word)
    word = re.sub(r'http\S+', '', word)
    word = re.sub("[^a-z0-9]"," ", word)
    word = re.sub('\[.*?\]',' ', word)
    word = re.sub(r'(-|\')', '', word)
    return word


# In[23]:


combi['tidy_tweet'] = combi['text'].apply(preprocess_word)
combi.head()


# ## Stop word removal

# In[24]:


stopwords = nltk.corpus.stopwords.words('english')
combi['tidy_tweet'].apply(lambda x: [item for item in x if item not in stopwords])


# In[25]:


combi.head()


# ## Handle Emojis

# In[26]:


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


# In[27]:


combi['tidy_tweet'] = combi['tidy_tweet'].apply(handle_emojis)
combi.head()


# ## Fix Bad Unicode

# In[28]:


def fix_bad_unicode(text, normalization="NFC"):
    try:
        text = text.encode("latin", "backslashreplace").decode("unicode-escape")
    except:
        pass

    return fix_text(text, normalization=normalization)


# In[29]:


combi['tidy_tweet'] = combi['tidy_tweet'].apply(handle_emojis)
combi.head()


# ## Remove Pattern

# In[30]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# In[31]:


combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tidy_tweet'], "@[\w]*") 
combi.head()


# ## Wordclouds

# In[32]:


all_words = ' '.join([text for text in combi['tidy_tweet']]) 
# fig = plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off')
plt.title('Dataset Word Clouds', fontsize=20)
# plt.show()
plt.tight_layout()
plt.savefig("Word_Clouds.png", dpi=200, bbox_inches='tight')


# normal_words =' '.join([text for text in combi['tidy_tweet'][combi['sentiment'] == 1]]) 

# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.title('Neutral Word Clouds', fontsize=20)
# # plt.show()
# plt.tight_layout()
# plt.savefig("Neutral_Word_Clouds.png", dpi=200, bbox_inches='tight')


# ## Neutral Sentiment Words

# In[33]:


normal_words =' '.join([text for text in combi['tidy_tweet'][combi['sentiment'] == 1]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Neutral Word Clouds', fontsize=20)
# plt.show()
plt.tight_layout()
plt.savefig("Neutral_Word_Clouds.png", dpi=200, bbox_inches='tight')


# ## Positive Sentiment Words

# In[34]:


normal_words =' '.join([text for text in combi['tidy_tweet'][combi['sentiment'] == 2]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Positive Word Clouds', fontsize=20)
# plt.show()
plt.tight_layout()
plt.savefig("Positive_Word_Clouds.png", dpi=200, bbox_inches='tight')


# In[35]:


combi['sentiment'].value_counts()


# ## Negative Sentiment Words

# In[36]:


normal_words =' '.join([text for text in combi['tidy_tweet'][combi['sentiment'] == 0]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Negative Word Clouds', fontsize=20)
# plt.show()
plt.tight_layout()
plt.savefig("Negative_Word_Clouds.png", dpi=200, bbox_inches='tight')


# ## Impact on Hashtags

# In[37]:


def hashtag_extract(x):
    hashtags = []    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


# In[38]:


HT_neutral = hashtag_extract(combi['text'][combi['sentiment'] == 1])


# In[39]:


HT_positive = hashtag_extract(combi['text'][combi['sentiment'] == 2]) 


# In[40]:


HT_negative = hashtag_extract(combi['text'][combi['sentiment'] == 0])


# In[41]:


HT_neutral = sum(HT_neutral,[]) 
HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])


# In[42]:


len(HT_negative)


# ## Neutral Sentiment Hashtags

# In[43]:


a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame(
    {
    'Hashtag': list(a.keys()),
    'Count': list(a.values())
    }
)


# In[44]:


d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(10,7))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.xticks(rotation=90)
plt.title('Neutral Words HashTags', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig("Neutral_Word_Hashtags.png", dpi=200, bbox_inches='tight')


# ## Positive Sentiment Hashtags

# In[45]:


a = nltk.FreqDist(HT_positive)
d = pd.DataFrame(
    {
    'Hashtag': list(a.keys()),
    'Count': list(a.values())
    }
)


# In[46]:


d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(10,7))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.xticks(rotation=90)
plt.title('Positive Words HashTags', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig("Positive_Word_Hashtags.png", dpi=200, bbox_inches='tight')


# ## Negative Sentiment Hashtags

# In[47]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame(
    {
    'Hashtag': list(a.keys()),
    'Count': list(a.values())
    }
)


# In[48]:


d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(10,7))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
plt.xticks(rotation=90)
plt.title('Negative Words HashTags', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig("Negative_Word_Hashtags.png", dpi=200, bbox_inches='tight')


# ## Bag-of-Words Features

# ## Build a Classification Model

# In[49]:


bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape


# ## TF-IDF Features

# In[50]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape


# ## Word2Vec Features

# In[51]:


get_ipython().run_cell_magic('time', '', "\ntokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing \n\nmodel_w2v = gensim.models.Word2Vec(\n            tokenized_tweet,\n            vector_size=500, # desired no. of features/independent variables\n            window=5, # context window size\n            min_count=2, # Ignores all words with total frequency lower than 2.                                  \n            sg = 1, # 1 for skip-gram model\n            hs = 0,\n            negative = 10, # for negative sampling\n            workers= 32, # no.of cores\n            seed = 34\n)\n\nmodel_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)")


# In[52]:


model_w2v.wv.most_similar(positive="fever")


# In[53]:


model_w2v.wv.most_similar(positive="throat")


# In[54]:


model_w2v.wv['moderna']


# In[55]:


len(model_w2v.wv['moderna'])


# ## Preparing Vectors for Tweets

# In[56]:


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


# ## Preparing Word2Vec Feature Set

# In[57]:


wordvec_arrays = np.zeros((len(tokenized_tweet), 500)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 500)
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape


# ## Doc2Vec Embedding

# In[58]:


def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output

labeled_tweets = add_label(tokenized_tweet) # label all the tweets


# In[59]:


labeled_tweets[:6]


# In[60]:


get_ipython().run_cell_magic('time', '', "model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model\n                                  dm_mean=1, # dm_mean = 1 for using mean of the context word vectors\n                                  vector_size=500, # no. of desired features\n                                  window=5, # width of the context window                                  \n                                  negative=7, # if > 0 then negative sampling will be used\n                                  min_count=5, # Ignores all words with total frequency lower than 5.                                  \n                                  workers=32, # no. of cores                                  \n                                  alpha=0.1, # learning rate                                  \n                                  seed = 23, # for reproducibility\n                                 ) \n\nmodel_d2v.build_vocab([i for i in tqdm(labeled_tweets)])\n\nmodel_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)")


# In[61]:


docvec_arrays = np.zeros((len(tokenized_tweet), 500)) 
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,500))    

docvec_df = pd.DataFrame(docvec_arrays) 
docvec_df.shape


# In[62]:


# Extracting train and test BoW features 
train_bow = bow[:16180,:]

# splitting data into training and validation set 
x_train,x_test,y_train,y_test = train_test_split(train_bow, combi['sentiment'], random_state=42, test_size=0.3)


# In[63]:


type(train_bow)


# ## Machine Learning for CounterVectorization

# ## SVC

# In[64]:


svc_model = LinearSVC(class_weight='balanced',C=1, penalty='l2', max_iter=1500,loss='squared_hinge',
                        multi_class='ovr').fit(x_train, y_train)

svc_model_predict = svc_model.predict(x_test)
svc_report = classification_report(y_test, svc_model_predict )

print(svc_report)


# ## SGDClassifier

# In[65]:


sgd_model = SGDClassifier(n_jobs=-1,class_weight='balanced',penalty='l2').fit(x_train, y_train)
sgd_model_predict = sgd_model.predict(x_test)
sgd_report = classification_report(y_test, sgd_model_predict )

print(sgd_report)


# ## MLPClassifier

# In[66]:


mlp = MLPClassifier().fit(x_train, y_train)
mlp_predict = mlp.predict(x_test)
mlp_report = classification_report(y_test, mlp_predict )
print(mlp_report)


# ## KNeighborsClassifier

# In[67]:


classifier = KNeighborsClassifier(n_neighbors=5,algorithm='brute') 
classifier.fit(x_train, y_train) 
predicted_label = classifier.predict(x_test) 
knn_model_report = classification_report(y_test, predicted_label)
print(knn_model_report)


# ## RandomForestClassifier

# In[68]:


rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
rf_predict = rf.predict(x_test)
rf_report = classification_report(y_test, rf_predict )
print(rf_report)


# ## AdaBoostClassifier

# In[69]:


ab = AdaBoostClassifier(n_estimators=100, learning_rate=1).fit(x_train, y_train)
ab_predict = ab.predict(x_test)
ab_report = classification_report(y_test, ab_predict )
print(ab_report)


# ## BaggingClassifier

# In[70]:


bagging = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(x_train, y_train)
bagging_predict = bagging.predict(x_test)
bagging_report = classification_report(y_test, bagging_predict )
print(bagging_report)


# ## ExtraTreesClassifier

# In[71]:


et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
et_predict = et.predict(x_test)
et_report = classification_report(y_test, et_predict)
print(et_report)


# ## DecisionTreeClassifier

# In[72]:


clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
dt_model_report = classification_report(y_test, y_pred)
print(dt_model_report)


# ## Logistic Regression

# In[73]:


logistic_reg_model = LogisticRegression(n_jobs = -1, penalty='l2', multi_class='multinomial',class_weight = 'balanced',verbose=1).fit(x_train,y_train)

lr_model_predict = logistic_reg_model.predict(x_test)
lr_model_report = classification_report(y_test, lr_model_predict)

print(lr_model_report)


# ## Padding and Tokenization

# In[74]:


x = combi.tidy_tweet
y = combi['sentiment']


# In[75]:


print(x.shape)
print(y.shape)


# In[76]:


num_words = 8000
embed_dim = 32
tokenizer = Tokenizer(num_words=num_words,oov_token = "<oov>" )
tokenizer.fit_on_texts(x)
word_index=tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)
length=[]
for i in sequences:
    length.append(len(i))
print(len(length))
print("Mean is: ",np.mean(length))
print("Max is: ",np.max(length))
print("Min is: ",np.min(length))


# In[77]:


pad_length = 24
sequences = pad_sequences(sequences, maxlen = pad_length, truncating = 'pre', padding = 'post')
sequences.shape


# In[78]:


x_train,x_test,y_train,y_test = train_test_split(sequences,y,test_size = 0.05)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Training Deep Learning Model and Embeddings

# In[79]:


recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()

model = Sequential([Embedding(num_words, embed_dim, input_length = pad_length),
                   SimpleRNN(8, return_sequences = True),
                   GlobalMaxPool1D(),
                   Dense(20,activation = 'relu',kernel_initializer='he_uniform'),
                   Dropout(0.25),
                   Dense(3,activation = 'softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.name = 'Twitter Hate Text Classification'
model.summary()


# In[80]:


history = model.fit(x = x_train, y = y_train, epochs = 5,validation_split = 0.05)


# In[81]:


train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# In[82]:


fig=plt.figure(figsize=(10,10))
fig.add_subplot(2, 1, 1)
plt.grid()
plt.plot(train_loss, color='blue', label='Train')
plt.plot(val_loss, color='red', label='Validation')
plt.legend()
plt.title('Loss Counter Vectorizer', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Loss_CounterVectorizer.png", dpi=200, bbox_inches='tight')


fig.add_subplot(2, 1, 2)
plt.grid()
plt.plot(train_acc, color='blue', label='Train')
plt.plot(valid_acc, color='red', label='Validation')
plt.legend()
plt.title('Classification Accuracy Counter Vectorizer', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Classifier_Accuracy_CounterVectorizer.png", dpi=200, bbox_inches='tight')


# In[83]:


evaluate = model.evaluate(x_test,y_test)


# In[84]:


print("Test Acuracy is : {:.2f} %".format(evaluate[1]*100))
print("Test Loss is : {:.4f}".format(evaluate[0]))


# In[85]:


predictions = model.predict(x_test)


# In[86]:


predict = []
for i in predictions:
    predict.append(np.argmax(i))


# In[87]:


cm = confusion_matrix(predict,y_test)
acc = accuracy_score(predict,y_test)


# In[88]:


print("The Confusion matrix is: \n",cm)


# In[89]:


print(acc*100)


# In[90]:


print(metrics.classification_report(y_test, predict))


# # Machine Learning for TF_IDF
# ## TF-IDF Features

# In[91]:


train_tfidf = tfidf[:16180,:]

x_train,x_test,y_train,y_test = train_test_split(train_tfidf, combi['sentiment'], random_state=42, test_size=0.3) 

xtrain_tfidf = train_tfidf[y_train.index]
xvalid_tfidf = train_tfidf[y_test.index]


# ## SVC

# In[92]:


svc_model = LinearSVC(class_weight='balanced',C=1, penalty='l2', max_iter=1500,loss='squared_hinge',
                        multi_class='ovr').fit(x_train, y_train)

svc_model_predict = svc_model.predict(x_test)
svc_report = classification_report(y_test, svc_model_predict )

print(svc_report)


# ## SGD Classifier

# In[93]:


sgd_model = SGDClassifier(n_jobs=-1,class_weight='balanced',penalty='l2').fit(x_train, y_train)
sgd_model_predict = sgd_model.predict(x_test)
sgd_report = classification_report(y_test, sgd_model_predict )

print(sgd_report)


# ## MLP Classifier

# In[94]:


mlp = MLPClassifier().fit(x_train, y_train)
mlp_predict = mlp.predict(x_test)
mlp_report = classification_report(y_test, mlp_predict )
print(mlp_report)


# ## KNeighbors Classifier

# In[95]:


classifier = KNeighborsClassifier(n_neighbors=5,algorithm='brute') 
classifier.fit(x_train, y_train) 
predicted_label = classifier.predict(x_test) 
knn_model_report = classification_report(y_test, predicted_label)
print(knn_model_report)


# ## Random Forest Classifier

# In[96]:


rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
rf_predict = rf.predict(x_test)
rf_report = classification_report(y_test, rf_predict )
print(rf_report)


# ## AdaBoost Classifier

# In[97]:


ab = AdaBoostClassifier(n_estimators=100, learning_rate=1).fit(x_train, y_train)
ab_predict = ab.predict(x_test)
ab_report = classification_report(y_test, ab_predict )
print(ab_report)


# ## Bagging Classifier

# In[98]:


bagging = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(x_train, y_train)
bagging_predict = bagging.predict(x_test)
bagging_report = classification_report(y_test, bagging_predict )
print(bagging_report)


# ## ExtraTrees Classifier

# In[99]:


et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
et_predict = et.predict(x_test)
et_report = classification_report(y_test, et_predict)
print(et_report)


# ## DecisionTree Classifier

# In[100]:


clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
dt_model_report = classification_report(y_test, y_pred)
print(dt_model_report)


# ## Logistic Regression

# In[101]:


logistic_reg_model = LogisticRegression(n_jobs = -1, penalty='l2', multi_class='multinomial',class_weight = 'balanced',verbose=1).fit(x_train,y_train)

lr_model_predict = logistic_reg_model.predict(x_test)
lr_model_report = classification_report(y_test, lr_model_predict)

print(lr_model_report)


# ## Padding and Tokenizing for TF-IDF

# In[102]:


num_words = 8000
embed_dim = 32
tokenizer = Tokenizer(num_words=num_words,oov_token = "<oov>" )
tokenizer.fit_on_texts(x)
word_index=tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)
length=[]
for i in sequences:
    length.append(len(i))
print(len(length))
print("Mean is: ",np.mean(length))
print("Max is: ",np.max(length))
print("Min is: ",np.min(length))


# In[103]:


pad_length = 24
sequences = pad_sequences(sequences, maxlen = pad_length, truncating = 'pre', padding = 'post')
sequences.shape


# In[104]:


x_train,x_test,y_train,y_test = train_test_split(sequences,y,test_size = 0.05)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Training Deep Learning Model on Embedding for TF-IDF

# In[105]:


recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()

model = Sequential([Embedding(num_words, embed_dim, input_length = pad_length),
                   SimpleRNN(8, return_sequences = True),
                   GlobalMaxPool1D(),
                   Dense(20,activation = 'relu',kernel_initializer='he_uniform'),
                   Dropout(0.25),
                   Dense(3,activation = 'softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.name = 'Twitter Hate Text Classification'
model.summary()


# In[106]:


history = model.fit(x = x_train, y = y_train, epochs = 5,validation_split = 0.05)


# In[107]:


train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# In[108]:


fig=plt.figure(figsize=(10,10))
fig.add_subplot(2, 1, 1)
plt.grid()
plt.plot(train_loss, color='blue', label='Train')
plt.plot(val_loss, color='red', label='Validation')
plt.legend()
plt.title('Loss TF-IDF', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Loss_TF_IDF.png", dpi=200, bbox_inches='tight')


fig.add_subplot(2, 1, 2)
plt.grid()
plt.plot(train_acc, color='blue', label='Train')
plt.plot(valid_acc, color='red', label='Validation')
plt.legend()
plt.title('Classification Accuracy TF-IDF', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Classifier_Accuracy_TF_IDF.png", dpi=200, bbox_inches='tight')


# In[109]:


evaluate = model.evaluate(x_test,y_test)
print("Test Acuracy is : {:.2f} %".format(evaluate[1]*100))
print("Test Loss is : {:.4f}".format(evaluate[0]))
predictions = model.predict(x_test)
predict = []
for i in predictions:
    predict.append(np.argmax(i))
    
cm = confusion_matrix(predict,y_test)
acc = accuracy_score(predict,y_test)

print("The Confusion matrix is: \n",cm)
print(acc*100)
print(metrics.classification_report(y_test, predict))


# # Machine learning for Word2Vec
# ## Word2vec Features

# In[110]:


train_w2v = wordvec_df.iloc[:16180,:]
x_train,x_test,y_train,y_test = train_test_split(train_w2v, combi['sentiment'], random_state=42, test_size=0.3)
xtrain_w2v = train_w2v.iloc[y_train.index,:]
xvalid_w2v = train_w2v.iloc[y_test.index,:]


# ## SVC

# In[111]:


svc_model = LinearSVC(class_weight='balanced',C=1, penalty='l2', max_iter=1500,loss='squared_hinge',
                        multi_class='ovr').fit(x_train, y_train)

svc_model_predict = svc_model.predict(x_test)
svc_report = classification_report(y_test, svc_model_predict )

print(svc_report)


# ## SGD Classifier

# In[112]:


sgd_model = SGDClassifier(n_jobs=-1,class_weight='balanced',penalty='l2').fit(x_train, y_train)
sgd_model_predict = sgd_model.predict(x_test)
sgd_report = classification_report(y_test, sgd_model_predict )

print(sgd_report)


# ## MLP Classifier

# In[113]:


mlp = MLPClassifier().fit(x_train, y_train)
mlp_predict = mlp.predict(x_test)
mlp_report = classification_report(y_test, mlp_predict )
print(mlp_report)


# ## KNeighbors Classifier

# In[114]:


classifier = KNeighborsClassifier(n_neighbors=5,algorithm='brute') 
classifier.fit(x_train, y_train) 
predicted_label = classifier.predict(x_test) 
knn_model_report = classification_report(y_test, predicted_label)
print(knn_model_report)


# ## RandomForest Classifier

# In[115]:


rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
rf_predict = rf.predict(x_test)
rf_report = classification_report(y_test, rf_predict )
print(rf_report)


# ## AdaBoost Classifier

# In[116]:


ab = AdaBoostClassifier(n_estimators=100, learning_rate=1).fit(x_train, y_train)
ab_predict = ab.predict(x_test)
ab_report = classification_report(y_test, ab_predict )
print(ab_report)


# ## Bagging Classifier

# In[117]:


bagging = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(x_train, y_train)
bagging_predict = bagging.predict(x_test)
bagging_report = classification_report(y_test, bagging_predict )
print(bagging_report)


# ## ExtraTrees Classifier

# In[118]:


et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
et_predict = et.predict(x_test)
et_report = classification_report(y_test, et_predict)
print(et_report)


# ## DecisionTree Classifier

# In[119]:


clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
dt_model_report = classification_report(y_test, y_pred)
print(dt_model_report)


# ## Logistic Regression

# In[120]:


logistic_reg_model = LogisticRegression(n_jobs = -1, penalty='l2', multi_class='multinomial',class_weight = 'balanced',verbose=1).fit(x_train,y_train)

lr_model_predict = logistic_reg_model.predict(x_test)
lr_model_report = classification_report(y_test, lr_model_predict)

print(lr_model_report)


# ## Padding and Tokenization for Word2Vec

# In[121]:


num_words = 8000
embed_dim = 32
tokenizer = Tokenizer(num_words=num_words,oov_token = "<oov>" )
tokenizer.fit_on_texts(x)
word_index=tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)
length=[]
for i in sequences:
    length.append(len(i))
print(len(length))
print("Mean is: ",np.mean(length))
print("Max is: ",np.max(length))
print("Min is: ",np.min(length))


# In[122]:


pad_length = 24
sequences = pad_sequences(sequences, maxlen = pad_length, truncating = 'pre', padding = 'post')
sequences.shape

x_train,x_test,y_train,y_test = train_test_split(sequences,y,test_size = 0.05)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Training Deep Learning Model on Embeddings for Word2vec

# In[123]:


recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()

model = Sequential([Embedding(num_words, embed_dim, input_length = pad_length),
                   SimpleRNN(8, return_sequences = True),
                   GlobalMaxPool1D(),
                   Dense(20,activation = 'relu',kernel_initializer='he_uniform'),
                   Dropout(0.25),
                   Dense(3,activation = 'softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.name = 'Twitter Hate Text Classification'
model.summary()


# In[124]:


history = model.fit(x = x_train, y = y_train, epochs = 5,validation_split = 0.05)

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# In[125]:


fig=plt.figure(figsize=(10,10))
fig.add_subplot(2, 1, 1)
plt.grid()
plt.plot(train_loss, color='blue', label='Train')
plt.plot(val_loss, color='red', label='Validation')
plt.legend()
plt.title('Loss Word2Vec', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Loss_Word2Vec.png", dpi=200, bbox_inches='tight')


fig.add_subplot(2, 1, 2)
plt.grid()
plt.plot(train_acc, color='blue', label='Train')
plt.plot(valid_acc, color='red', label='Validation')
plt.legend()
plt.title('Classification Accuracy Word2Vec', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Classifier_Accuracy_Word2Vec.png", dpi=200, bbox_inches='tight')


# In[126]:


evaluate = model.evaluate(x_test,y_test)
print("Test Acuracy is : {:.2f} %".format(evaluate[1]*100))
print("Test Loss is : {:.4f}".format(evaluate[0]))
predictions = model.predict(x_test)
predict = []
for i in predictions:
    predict.append(np.argmax(i))
    
cm = confusion_matrix(predict,y_test)
acc = accuracy_score(predict,y_test)

print("The Confusion matrix is: \n",cm)
print(acc*100)
print(metrics.classification_report(y_test, predict))


# # Machine Learning for Doc2Vec
# ## Doc2Vec Features

# In[127]:


train_d2v = docvec_df.iloc[:16180,:]
x_train,x_test,y_train,y_test = train_test_split(train_d2v, combi['sentiment'], random_state=42, test_size=0.3)
xtrain_d2v = train_d2v.iloc[y_train.index,:]
xvalid_d2v = train_d2v.iloc[y_test.index,:]


# ## SVC

# In[128]:


svc_model = LinearSVC(class_weight='balanced',C=1, penalty='l2', max_iter=1500,loss='squared_hinge',
                        multi_class='ovr').fit(x_train, y_train)

svc_model_predict = svc_model.predict(x_test)
svc_report = classification_report(y_test, svc_model_predict )

print(svc_report)


# ## SGD Classifier

# In[129]:


sgd_model = SGDClassifier(n_jobs=-1,class_weight='balanced',penalty='l2').fit(x_train, y_train)
sgd_model_predict = sgd_model.predict(x_test)
sgd_report = classification_report(y_test, sgd_model_predict )

print(sgd_report)


# ## MLP Classifier

# In[130]:


mlp = MLPClassifier().fit(x_train, y_train)
mlp_predict = mlp.predict(x_test)
mlp_report = classification_report(y_test, mlp_predict )
print(mlp_report)


# ## KNeighbors Classifier

# In[131]:


classifier = KNeighborsClassifier(n_neighbors=5,algorithm='brute') 
classifier.fit(x_train, y_train) 
predicted_label = classifier.predict(x_test) 
knn_model_report = classification_report(y_test, predicted_label)
print(knn_model_report)


# ## RandomForest Classifier

# In[132]:


rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
rf_predict = rf.predict(x_test)
rf_report = classification_report(y_test, rf_predict )
print(rf_report)


# ## AdaBoost Classifier

# In[133]:


ab = AdaBoostClassifier(n_estimators=100, learning_rate=1).fit(x_train, y_train)
ab_predict = ab.predict(x_test)
ab_report = classification_report(y_test, ab_predict )
print(ab_report)


# ## Bagging Classifier

# In[134]:


bagging = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(x_train, y_train)
bagging_predict = bagging.predict(x_test)
bagging_report = classification_report(y_test, bagging_predict )
print(bagging_report)


# ## ExtraTrees Classifier

# In[135]:


et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
et_predict = et.predict(x_test)
et_report = classification_report(y_test, et_predict)
print(et_report)


# ## DecisionTree Classifier

# In[136]:


clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
dt_model_report = classification_report(y_test, y_pred)
print(dt_model_report)


# ## Logistic Regression

# In[137]:


logistic_reg_model = LogisticRegression(n_jobs = -1, penalty='l2', multi_class='multinomial',class_weight = 'balanced',verbose=1).fit(x_train,y_train)

lr_model_predict = logistic_reg_model.predict(x_test)
lr_model_report = classification_report(y_test, lr_model_predict)

print(lr_model_report)


# ## Padding and Tokenzation Doc2Feature

# In[138]:


num_words = 8000
embed_dim = 32
tokenizer = Tokenizer(num_words=num_words,oov_token = "<oov>" )
tokenizer.fit_on_texts(x)
word_index=tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)
length=[]
for i in sequences:
    length.append(len(i))
print(len(length))
print("Mean is: ",np.mean(length))
print("Max is: ",np.max(length))
print("Min is: ",np.min(length))

pad_length = 24
sequences = pad_sequences(sequences, maxlen = pad_length, truncating = 'pre', padding = 'post')
sequences.shape

x_train,x_test,y_train,y_test = train_test_split(sequences,y,test_size = 0.05)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Training Deep Learning Model on Embeddings Doc2Feature

# In[139]:


recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()

model = Sequential([Embedding(num_words, embed_dim, input_length = pad_length),
                   SimpleRNN(8, return_sequences = True),
                   GlobalMaxPool1D(),
                   Dense(20,activation = 'relu',kernel_initializer='he_uniform'),
                   Dropout(0.25),
                   Dense(3,activation = 'softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.name = 'Twitter Hate Text Classification'
model.summary()


# In[140]:


history = model.fit(x = x_train, y = y_train, epochs = 5,validation_split = 0.05)

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# In[141]:


fig=plt.figure(figsize=(10,10))
fig.add_subplot(2, 1, 1)
plt.grid()
plt.plot(train_loss, color='blue', label='Train')
plt.plot(val_loss, color='red', label='Validation')
plt.legend()
plt.title('Loss Doc2Feature', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Loss_Doc2Feature.png", dpi=200, bbox_inches='tight')


fig.add_subplot(2, 1, 2)
plt.grid()
plt.plot(train_acc, color='blue', label='Train')
plt.plot(valid_acc, color='red', label='Validation')
plt.legend()
plt.title('Classification Accuracy Doc2Feature', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig("Classifier_Accuracy_Doc2Feature.png", dpi=200, bbox_inches='tight')


# In[142]:


evaluate = model.evaluate(x_test,y_test)
print("Test Acuracy is : {:.2f} %".format(evaluate[1]*100))
print("Test Loss is : {:.4f}".format(evaluate[0]))
predictions = model.predict(x_test)
predict = []
for i in predictions:
    predict.append(np.argmax(i))
    
cm = confusion_matrix(predict,y_test)
acc = accuracy_score(predict,y_test)

print("The Confusion matrix is: \n",cm)
print(acc*100)
print(metrics.classification_report(y_test, predict))


# In[ ]:




