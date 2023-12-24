#!/usr/bin/env python
# coding: utf-8

# In[859]:


import pandas_profiling
import nltk
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt 
import seaborn as sb
from nltk.corpus import stopwords
import warnings 
warnings.filterwarnings("ignore")
import unidecode
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
from nltk.stem import PorterStemmer
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
import matplotlib.animation as animation
import operator
import plotly.express as px
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
import io
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[860]:


df = pd.read_csv('train.csv')
df.head()


# In[861]:


#pandas_profiling.ProfileReport(df) #checking for any duplicate values or missing values


# In[862]:


df.shape


# In[863]:


df['clean_tweet'] = df['tweet'].apply(lambda x : ' '.join([tweet for tweet in x.split()if not tweet.startswith("@")]))#removes the words with @
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([tweet for tweet in x.split() if not tweet == '\d*']))#removes numbers
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([unidecode.unidecode(word) for word in x.split()])) #removes other than english
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([word for word in x.split() if not word == 'h(m)+' ]))#removing words with hmm
p=[]
'''def preprocessing(text):
    for sentence in tqdm(text):
        t=re.sub("[^A-Za-z0-9]+"," ",sentence)
        p.append(t)
    return p
    
preprocessed_tweets=preprocessing(df['tweet'].values)

df['clean_tweet']=preprocessed_tweets'''
df.head(10)


# In[864]:


d = {'luv':'love','wud':'would','lyk':'like','wateva':'whatever','ttyl':'talk to you later',
               'kul':'cool','fyn':'fine','omg':'oh my god!','fam':'family','bruh':'brother','u':'you',
               'cud':'could','fud':'food',"can't":'cannot',"won't":'will not','re':'are','m':'am','ll':'will','ve':'have','nt':'not',} #short notations to full form

df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join(d[word] if word in d else word for word in x.split()))
df.head(10)


# In[865]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([word for word in x.split() if not word in set(stopwords.words('english'))]))
df.head(10)


# In[866]:


lemmatizer = WordNetLemmatizer()
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
df.head(10)


# In[867]:


ps = PorterStemmer()
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split()]))
df.head(10)


# In[868]:


corpus = []
for i in range(0,31962):
    tweet = df['clean_tweet'][i]
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    tweet = re.sub("#","", tweet)
    corpus.append(tweet)


# In[869]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([re.sub("#","",word) for word in x.split()]))
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([re.sub("[^A-Za-z0-9]+","",word) for word in x.split()]))
#len(corpus)
#corpus
df.head()


# In[870]:


df=df.iloc[:,[1,3]]
df.head()


# In[871]:


df.shape


# In[872]:


dff = df.drop(['label'], axis = 1)
dff.head()


# In[873]:


t=open("top_5k_twitter_2015.txt",'r',encoding='utf8') ; d=t.read()
c=0;
normal_words = ' '.join([word for word in df['clean_tweet'][df['label'] == 1]])
l=list(normal_words.split(" "))
l=list(set(l))
for i in l:
    if i in d:
        c=c+1
print(len(normal_words))
print(c)
wordcloud = WordCloud(width = 800, height = 500, max_font_size = 110,max_words = 300).generate(normal_words)
print('Normal words')
plt.figure(figsize= (12,8))
plt.imshow(wordcloud, interpolation = 'bilinear',cmap='viridis')
plt.axis('off')


# In[874]:


from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(dff, list(df.label), test_size=0.1)


# In[875]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[876]:


vect=TfidfVectorizer(min_df=10)

vect.fit(X_temp['clean_tweet'].values)

train_tweet=vect.transform(X_temp['clean_tweet'].values)
test_tweet=vect.transform(X_test['clean_tweet'].values)

#print(train_tweet.shape,y_temp.shape)
#print(test_tweet.shape,y_test.shape)


# In[877]:


x_train_sent=np.ndarray.tolist(X_temp["clean_tweet"].values)

sia=SentimentIntensityAnalyzer()
ps=[]
for i in range(len(x_train_sent)):
    ps.append((sia.polarity_scores((x_train_sent[i]))))
    
x_train_polarity=np.array(ps)
x_train_polarity=x_train_polarity.reshape(-1,1)
x_train_polarity.shape


# In[878]:


x_t=[]
for i in range(len(X_temp)):
    for j in x_train_polarity[0][0]:
        x_t.append(x_train_polarity[i][0][j])
x_t=np.array(x_t)
x_t=x_t.reshape(-1,4)
x_t.shape


# In[879]:


x_test_sent=np.ndarray.tolist(X_test["clean_tweet"].values)

sia=SentimentIntensityAnalyzer()
ps=[]
for i in range(len(x_test_sent)):
    ps.append((sia.polarity_scores((x_test_sent[i]))))
    
x_test_polarity=np.array(ps)
x_test_polarity=x_test_polarity.reshape(-1,1)
x_test_polarity.shape


# In[880]:


x_tests=[]
for i in range(len(X_test)):
    for j in x_test_polarity[0][0]:
        x_tests.append(x_test_polarity[i][0][j])
x_tests=np.array(x_tests)
x_tests=x_tests.reshape(-1,4)
x_tests.shape


# In[881]:


from scipy.sparse import hstack

x_tr=hstack((train_tweet,x_t))
x_te=hstack((test_tweet,x_tests))



print(x_tr.shape)
print(x_te.shape)


# In[882]:


print(test_tweet)


# In[883]:


wt={0:1,1:5}            #since the data is imbalanced , we assign some more weight to class 1

clf=DecisionTreeClassifier(class_weight=wt)

parameters=dict(max_depth=[1,5,10,50],min_samples_split=[5,10,100,500])

search=RandomizedSearchCV(clf,parameters,random_state=10)
result=search.fit(x_tr,y_temp)
result.cv_results_


# In[884]:


cls = DecisionTreeClassifier(max_depth=50,min_samples_split=5,random_state=10,class_weight=wt)
cls.fit(x_tr,y_temp)
y_pred_train=cls.predict(x_tr)
y_pred_test=cls.predict(x_te)

train_fpr,train_tpr,tr_treshold=roc_curve(y_temp,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve")
plt.grid()
plt.show()


# In[885]:


def find_best_threshold(threshold, fpr, tpr):
    """it will give best threshold value that will give the least fpr"""
    t = threshold[np.argmax(tpr*(1-fpr))]
    
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    
    return t

def predict_with_best_t(proba, threshold):
    """this will give predictions based on best threshold value"""
    predictions = []
    for i in proba:
        if i>=threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

#computing confusion matrix for set_1

from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_treshold, train_fpr, train_tpr)
print("Train confusion matrix")
m_tr=(confusion_matrix(y_temp, predict_with_best_t(y_pred_train, best_t)))
print(m_tr)
print("Test confusion matrix")
m_te=(confusion_matrix(y_test, predict_with_best_t(y_pred_test, best_t)))
print(m_te)


# In[886]:


print(classification_report(y_test, y_pred_test))


# In[887]:


len(y_test)


# In[888]:


X_temp['label'] = y_temp


# In[889]:


nonhate = X_temp[X_temp['label'] == 0]
nonhate.shape



hate = X_temp[X_temp['label'] == 1]

hate.shape[0]


# In[890]:


nonhatesample = nonhate.sample(n = hate.shape[0])
nonhatesample.shape


# In[891]:


hate.shape


# In[892]:


ds = pd.concat([hate, nonhatesample], axis = 0)
ds.head()
ds.tail()


# In[893]:


ds.shape


# In[894]:


ds.to_csv("trainset.csv")


# In[895]:


ds = pd.read_csv("trainset.csv",index_col=[0])


# In[117]:


ds.head()


# In[118]:


ds.head()


# In[119]:


ds_temp = ds
testdf = X_test
testdf['label'] = y_test


# In[120]:


testdf.shape


# In[121]:


ds = pd.concat([ds_temp, testdf], axis = 0)
ds.head()


# In[122]:


ds.shape


# In[123]:


testdf.head()


# In[124]:


list(testdf.index)


# In[125]:


corpus = []
for i in range(ds.shape[0]):
    corpus.append(ds.iloc[i][0])
corpus


# In[126]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
df2 = pd.DataFrame(denselist, columns=feature_names)
df2


# In[127]:


tdf = df2
tdf['labelxyz'] = list(ds.label)


# In[128]:


tdf.tail()


# In[129]:


tdf_hate = tdf[tdf.labelxyz == 1]
tdf_hate.shape


# In[130]:


tdf_nonhate = tdf[tdf.labelxyz == 0]
tdf_nonhate.shape


# In[131]:


X_train_hate = tdf_hate.sample(frac=0.9, random_state=0)
X_test_hate = tdf_hate.drop(X_train_hate.index)
X_train_nonhate =  tdf_nonhate.sample(frac=0.406, random_state=0)
X_test_nonhate = tdf_nonhate.drop(X_train_nonhate.index)


X_train_df = pd.concat([X_train_hate, X_train_nonhate], axis = 0)

X_train_df = shuffle(X_train_df)


# In[132]:


X_train = X_train_df.drop(['labelxyz'], axis = 1)
y_train = list(X_train_df.labelxyz)
X_test_df = pd.concat([X_test_hate, X_test_nonhate], axis = 0)

X_test = X_test_df.drop(['labelxyz'], axis = 1)
y_test = list(X_test_df.labelxyz)


# In[133]:


#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB


# In[134]:


# Use Cross-validation.
from sklearn.model_selection import cross_val_score

# Logistic Regression
'''log_reg = LogisticRegression()
log_scores = cross_val_score(log_reg, X_train, y_train, cv=3)
log_reg_mean = log_scores.mean()'''


# Naives Bayes
nav_clf = MultinomialNB()# works better on multinomial
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=3)
nav_mean = nav_scores.mean()

# Create a Dataframe with the results.
d = {'Classifiers': [ 'Naives Bayes'], 
    'Crossval Mean Scores': [ nav_mean]}

result_df = pd.DataFrame(data=d)


# In[135]:


result_df


# In[136]:


from sklearn.metrics import accuracy_score
nav_clf = MultinomialNB()
nav_clf.fit(X_train, y_train)
predict_nav = nav_clf.predict(X_test)
accuracy_score(y_test, predict_nav)


# In[137]:


from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(y_test, predict_nav)
print(cf_matrix)


# In[138]:


import seaborn as sns

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[139]:


from sklearn.metrics import roc_curve
nav_fpr, nav_tpr, threshold = roc_curve(y_test, predict_nav)
plt.plot(nav_fpr, nav_tpr, label='Naive Bayes')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),arrowprops=dict(facecolor='#6E726D', shrink=0.05),)
plt.legend()
plt.show()


# In[140]:


df1 = pd.DataFrame(classification_report((nav_clf.predict(X_test)[:] >= 0.3),  y_test, digits=2,output_dict=True)).T
#df1 = pd.DataFrame(classification_report(nav_clf.predict(X_test), y_test, digits=2,output_dict=True)).T


df1['support'] = df1.support.apply(int)

df1.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])


# In[141]:


clf=DecisionTreeClassifier()

parameters=dict(max_depth=[1,5,10,50],min_samples_split=[5,10,100,500])

search=RandomizedSearchCV(clf,parameters,random_state=10)
result=search.fit(X_train,y_train)
result.cv_results_

cls = DecisionTreeClassifier(max_depth=50,min_samples_split=5,random_state=10)
cls.fit(X_train,y_train)
y_pred_train=cls.predict(X_train)
y_pred_test=cls.predict(X_test)

print(accuracy_score(y_test,y_pred_test))


# In[66]:




#bow with top n words


# In[67]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus)
bow = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
bow['labelxyz'] = list(ds.label)

words = []
for k in range(len(corpus)):
    tweet_k = corpus[k].split(" ")
    for m in range(len(tweet_k)):
        words.append(tweet_k[m])


# In[68]:


from collections import Counter


counter_obj = Counter(words)
top100 = counter_obj.most_common(1000)
top100words = []
for i in range(len(top100)):
    top100words.append(top100[i][0])



lst3 = [value for value in list(bow.columns) if value in top100words] 

bow100 = bow[lst3]


# In[69]:


bow100['labelxyz'] = list(ds.label)
bow_hate = bow100[bow100.labelxyz == 1]
bow_nonhate = bow100[bow100.labelxyz == 0]
X_train_hate = bow_hate.sample(frac=0.9, random_state=0)
X_test_hate = bow_hate.drop(X_train_hate.index)
X_train_nonhate =  bow_nonhate.sample(frac=0.406, random_state=0)
X_test_nonhate = bow_nonhate.drop(X_train_nonhate.index)
X_train_df = pd.concat([X_train_hate, X_train_nonhate], axis = 0)
X_train = X_train_df.drop(['labelxyz'], axis = 1)
y_train = list(X_train_df.labelxyz)
X_test_df = pd.concat([X_test_hate, X_test_nonhate], axis = 0)
X_test = X_test_df.drop(['labelxyz'], axis = 1)
y_test = list(X_test_df.labelxyz)


# In[70]:


from sklearn.metrics import accuracy_score
nav_clf.fit(X_train, y_train)
predict_nav = nav_clf.predict(X_test)
print('naive bayes')
print(accuracy_score(y_test, predict_nav))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predict_nav))


# In[72]:


'''log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
predict_log = log_reg.predict(X_test)
accuracy_score(y_test, predict_log)'''


# In[73]:


from sklearn.metrics import roc_curve
nav_fpr, nav_tpr, threshold = roc_curve(y_test, predict_nav)
plt.plot(nav_fpr, nav_tpr, label='Naive Bayes')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),arrowprops=dict(facecolor='#6E726D', shrink=0.05),)
plt.legend()
plt.show()


# In[74]:


df1 = pd.DataFrame(classification_report(nav_clf.predict(X_test), 
                                        y_test, digits=2,
                                        output_dict=True)).T

df1['support'] = df1.support.apply(int)

df1.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])


# In[75]:



clf=DecisionTreeClassifier()

parameters=dict(max_depth=[1,5,10,50],min_samples_split=[5,10,100,500])

search=RandomizedSearchCV(clf,parameters,random_state=10)
result=search.fit(X_train,y_train)
result.cv_results_


# In[76]:


search.best_params_


# In[77]:


cls = DecisionTreeClassifier(max_depth=50,min_samples_split=5,random_state=10)
cls.fit(X_train,y_train)
y_pred_train=cls.predict(X_train)
y_pred_test=cls.predict(X_test)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve")
plt.grid()
plt.show()


# In[78]:


def find_best_threshold(threshold, fpr, tpr):
    """it will give best threshold value that will give the least fpr"""
    t = threshold[np.argmax(tpr*(1-fpr))]
    
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    
    return t

def predict_with_best_t(proba, threshold):
    """this will give predictions based on best threshold value"""
    predictions = []
    for i in proba:
        if i>=threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

#computing confusion matrix for set_1

from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_treshold, train_fpr, train_tpr)
print("Train confusion matrix")
m_tr=(confusion_matrix(y_train, predict_with_best_t(y_pred_train, best_t)))
print(m_tr)
print("Test confusion matrix")
m_te=(confusion_matrix(y_test, predict_with_best_t(y_pred_test, best_t)))
print(m_te)


# In[79]:


print(classification_report(y_test, y_pred_test))


# In[80]:


print(accuracy_score(y_test,y_pred_test))


# In[ ]:




