# Detecting-Hateful-Speech-in-Tweets-using-Sentiment-Analysis
Hate speech on social media has become a major issue in today’s world due to the increased usage of the internet and people’s different views. This freedom of speech is misused to direct hate towards individuals or groups based on their race, religion, gender, etc. Distinguishing toxic content is a key challenge. In this report, we propose an approach to automatically classify Twitter text into two classes as hate speech and non-hate speech.
## Table of Contents
* [Objectives]
* [Dataset Description]
* [Methodology]
    * [Data Preprocessing]
    * [Data Preparation]
* [Results]
## Objectives
* The main objectives of the project include:
* Analyzing a dataset from Kaggle containing Twitter text related to hate speech.
* Utilizing TFIDF and Bag-of-Words feature extraction techniques.
* Training multiple machine learning models, such as Naive Bayes and Decision Tree Classifier.
* Performing comparative analysis, including Vowpal Wabbit.
* Achieving a high level of accuracy in hate speech classification.

## Dataset Description
The data used for this project is taken from Kaggle, consisting of 31,935 tweets related to hate speech. The dataset is imbalanced, with 93% positive tweets and 7% negative tweets. The goal is to classify tweets into hate speech (1) or non-hate speech (0) using various feature extraction methods.

## Methodology:
### Data Preprocessing
* Cleaning tweets by removing mentions, numbers, and non-English words.
* Lemmatization and stemming to group variations of words.
* Substituting hashtag words without hashtags.

### Data Preparation
* Reducing the number of samples to balance positive and negative tweets.
* Feature extraction using TFIDF and Bag-of-Words.
* Training models, including Naive Bayes, Vowpal Wabbit, and Decision Tree.

## Results

After model tuning, the project achieved an accuracy of 88%. Evaluation metrics include accuracy, precision, and F1 score. The comparative analysis with Vowpal Wabbit demonstrated the effectiveness of the proposed approach.
