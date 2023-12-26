# Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis
Hate speech on social media has become a major issue in today’s world due to the increased usage of the internet and people’s different views. This freedom of speech is misused to direct hate towards individuals or groups based on their race, religion, gender, etc. Distinguishing toxic content is a key challenge. In this report, we propose an approach to automatically classify Twitter text into two classes as hate speech and non-hate speech.
## Table of Contents
* [Environment](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/blob/main/README.md#environment)
* [Objectives](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#objectives)
* [Dataset Description](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#dataset-description)
* [Methodology](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#methodology)
    * [Data Preprocessing](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#data-preprocessing)
    * [Data Preparation](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#data-preparation)
* [Results](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#results)
* [Developers](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis#developers)

## Environment:
- Jupyter Notebook on windows/Linux
- Libraries required to be installed:
   1. For code_ML.py: pandas, numpy, re, matplotlib, seaborn, nltk, operator, wordcloud, sklearn, unidecode
   2. For code_VW.py:vw, pandas,random




## Objectives
* The main objectives of the project include:
* Analyzing a dataset from Kaggle containing Twitter text related to hate speech.
* Utilizing TFIDF and Bag-of-Words feature extraction techniques.
* Training multiple machine learning models, such as Naive Bayes and Decision Tree Classifier.
* Performing comparative analysis, including Vowpal Wabbit.
* Achieving a high level of accuracy in hate speech classification.

## Dataset Description
The data used for this project is taken from Kaggle, consisting of 31,935 tweets related to hate speech. The dataset is imbalanced, with 93% positive tweets and 7% negative tweets. The goal is to classify tweets into hate speech (1) or non-hate speech (0) using various feature extraction methods.

## Methodology
### Data Preprocessing
* Cleaning tweets by removing mentions, numbers, and non-English words.
* Lemmatization and stemming to group variations of words.
* Substituting hashtag words without hashtags.

### Data Preparation
* Reducing the number of samples to balance positive and negative tweets.
* Feature extraction using TFIDF and Bag-of-Words.
* Training models, including Naive Bayes, Vowpal Wabbit, and Decision Tree.

## Results
![decision](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/assets/97566249/f72ef5dd-0962-4550-9f3f-fd3c27a33bde)

![naive](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/assets/97566249/2ecc55d4-cd1b-4ca3-a31d-bba9966c5f68)

![vowfe](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/assets/97566249/42d34e29-9fda-438f-a597-f60891e54933)

![vowte](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/assets/97566249/1449391b-d615-49f3-96ed-8cb49a91cf7a)

![vowtr](https://github.com/poornikabonam/Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis/assets/97566249/f21d5ed9-e960-41bb-95c7-aa4f55ef2be6)




After model tuning, the project achieved an accuracy of 88%. Evaluation metrics include accuracy, precision, and F1 score. The comparative analysis with Vowpal Wabbit demonstrated the effectiveness of the proposed approach.

## Developers
- [Poornika Bonam](https://github.com/poornikabonam)
- [Aditya Karnam](https://github.com/iamkarnam1999)
