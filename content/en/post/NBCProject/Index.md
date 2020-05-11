---
title: Term Project
subtitle: Use the [board game geek review data](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews) to design an algorithm model that can predict the rating for the given review.[kaggle](https://www.kaggle.com/yongli6/predictor-of-reviews).[Algorithm Git](https://github.com/VegetableCattle/youxinyu/tree/master/static/files/NBCproject.ipynb).[Download Algorithm file](https://yongli.netlify.com/files/NBCproject.ipynb). And then using this model to develop a predictor of reviews website. [Website Link](https://flask-rating-prediction.herokuapp.com/predicted). [Website Git](https://github.com/VegetableCattle/predictor-of-reviews).[demo readme](https://github.com/VegetableCattle/predictor-of-reviews/blob/master/README.md).[demo video](https://youtu.be/UXRi-gDB75g)
summary: Use the [board game geek review data](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews) to design an algorithm model that can predict the rating for the given  review.[Algorithm Git](https://github.com/VegetableCattle/youxinyu/tree/master/static/files/NBCproject.ipynb).[Download Algorithm file](https://yongli.netlify.com/files/NBCproject.ipynb). And then using this model to develop a predictor of reviews website. [Website Link](https://flask-rating-prediction.herokuapp.com/predicted). [Website Github](https://github.com/VegetableCattle/predictor-of-reviews).[demo readme](https://github.com/VegetableCattle/predictor-of-reviews/blob/master/README.md).[demo video](https://youtu.be/UXRi-gDB75g)
date: "2020-05-11T00:00:00Z"
math : true
comments: false
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""
  
---

## Blog content: Introduction the implementation of Naive Bayse Classifier without using algorithm library.

### 1.Algorithm principle

### 1.1 Introduction

The Bayesian method is a method with a long history and a lot of theoretical foundations. At the same time, it is direct and efficient when dealing with many problems. Many advanced natural language processing models can also evolve from it. Therefore, learning the Bayesian method is a very good entry point for studying natural language processing problems.

### 1.2 Bayesian formula

####  Joint probability formula: $ P(Y,X) = P(Y|X)P(X)=P(X|Y)P(Y) $

Among them, $P(Y) $ is called the prior probability, $P(Y∣X)$ is called the posterior probability, and $P(Y, X)$ is called the joint probability. This way we can derive the Bayesian formula.

#### Bayesian formula: $ P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)} $

### 1.3 Naive Bayes Algorithm

The probability model classifier is a conditional probability model:

$p(C|F_1,…,F_n) $

The independent variable C has several categories and the conditions depend on several feature variables, but the problem is that if the dimension of the number of features n is large or each feature can take a large number of values, it is not realistic to list the probability table based on the probability model. So we modified this model to make it feasible. According to the Bayesian formula:

$p(C│F_1,…,F_n )=\frac{p\(C)*p(F_1,…,F_n |C)}{p(F_1,…,F_n)} $

The denominator does not depend on C, and the value of the feature is also given, so the denominator can be considered a constant. The molecules are then equivalent to a joint distribution model:

$p(C|F_1,…,F_n) $

$∝p\(C)*p(F_1,…,F_n│C) $

$∝p\(C)*p(F_1│C)*p(F_2,…,F_n│C,F_1 ) $

$∝p\(C)*p(F_1│C)*p(F_2│C,F_1 )p(F_3│C,F_1,F_2 )…p(F_n│C,F_1,F_2…F_(n-1) )$

Assuming that each feature is independent of other features, that is, the features are independent of each other, there is:

$p(F_i│C,F_j )=p(F_i│C) $

This means that the conditional distribution of the variable C can be expressed as:

$p(C│F_1,…,F_n )=\frac{1}{Z} p\(C)*∏_i^np(F_i│C)$

The corresponding classifier is the formula defined as follows:

$classify(f_1,…,f_n )=argmax p(C=c)∏_i^np(F_i=f_i |C=c) $

In this model, I use the ratings as the class, the comments as object, the word of comments as feature.

### 2.Data Pre-processing

### 2.1 Data cleaning

First of all, this experiment only used the comment and rating columns in the dataset, so we filtered out the other columns. 

Second, we filter out the data with empty comment, because our purpose is to use the comment text to predict its rating.

Then we filter out comments in languages other than English.

### 2.2 Data Pre-processing

Replace the punctuation marks in the text with spaces to prepare for word segmentation later.

Because ratings are between 0-10 and are continuous values, we use the rounding method to convert these values to discrete values.

After many experiments, it is found that the accuracy rate obtained by using 0-10 integer scores is very low, probably based on between 20% and 30%, because for example, 9 points and 10 points are considered satisfactory scores, and the wording of comment is very similar. So I referred to the iPhone App Store and Amazon ’s 5-point rating standards, and by dividing this data by 2, the ratings data is mapped to 1-5 points.

### Some basic function libraries


```python
import os
import numpy as np
import re
import random
import pandas as pd
from csv import reader
```

### load data and data pre-processing


```python
def SegmentLineToWordsList(string):
    return list([x.lower() for x in re.split(r'[\s]\s*',string.strip()) if x])
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if row[3] == '':
                continue
            if row[2] == 'rating':
                continue
            sentense = re.sub("[%s]+"%('"|#|$|%|&|\|(|)|\[|\]|*|+|\-|/|<|=|>|@|^|`|{|}|~|_|,|.|?|!|:|;'), ' ', row[3])
            sentense = re.sub("[%s]+"%('\''),'',sentense)
            pattern = r'\w+'
            ascii_pattern = re.compile(pattern, re.ASCII)
            if len(ascii_pattern.findall(sentense)) == len(SegmentLineToWordsList(sentense)):
                index = round(float(row[2]) / 2)
                index = int(index)
                if index == 5:
                    index = 4
                dataset.append([sentense, index])
    return dataset

dataset_org = load_csv('./bgg-13m-reviews.csv')
print(len(dataset_org))
```

    2531939


### Split dataset to train set and test set


```python
def splitDataset(dataset, ratio_train):
    random.shuffle(dataset)
    cnt_train = round(len(dataset) * ratio_train ,0)
    train = []
    test = []
    for i in range(int(cnt_train)):
        train.append(dataset[i])
    for i in range(int(cnt_train) ,len(dataset)):
        test.append(dataset[i])
    return train, test

train = []
test = []
train, test = splitDataset(dataset_org, 0.75)
print(len(train))
print(len(test))
```

    1898954
    632985


### 3.Contributions & Optimization

### 3.1 implementation of Naive Bayse Classification without using algorithm library

### 3.2 Map ratings to 1-5 intervals

Through many experiments, it is found that the score is mapped to the 1-5 interval, and then the discretization is performed, and the final algorithm accuracy rate is nearly doubled.

### 3.3 Remove stop words

We observe **("This", "is", "a", "good","and", "nice", "game")** this sentence. In fact, words like **"This" and "is" are actually very neutral, no matter whether they appear in spam or not, they are not useful information to help judge. So we can directly ignore these typical words.** These words that are not helpful for our classification are called "Stop Words". This can reduce the time for us to train the model and judge the classification.

So the previous sentence becomes **("good", "nice", "game")**


```python
stopSet = set({'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'us', 'ourselves', 'you', 'your', 'yours', 
               'yourself', "youve", 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
               'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
               'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
               'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
               'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
               'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
               'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
               'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
               'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
               'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
               'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
               'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 
               'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 
               "havent", "wont", 'mustnt', "neednt", 'couldnt', 'doesnt', "shouldnt", "wasnt", 'wouldnt', "shes",
               "shouldve", "werent", "isnt", "dont", "arent", "thatll", "hasnt", "didnt", "mightnt", "hadnt", 'youre', 'theyre', })
```

### 3.4 Using inverted index dictionary to build vocabulary list

I use the data structure like: word:\[The number of occurrences in rating = 1,The number of occurrences in rating = 2,The number of occurrences in rating = 3,The number of occurrences in rating = 4,The number of occurrences in rating = 5\].This can increase the retrieval speed when using Bayesian algorithm.

### 3.5 Set the minimum threshold for the number of occurrences of words

Set the minimum threshold for the number of occurrences of words to eliminate the influence of rare words. After many experiments, the threshold value set as $len (train) * 0.00002$ .

### 3.6 Handling repeated words using a mixed model

#### 3.6.1 Polynomial model
If we consider the situation of repeated words, that is to say, the repeated words are regarded as their occurrence multiple times, and are directly derived according to the conditional independent assumption, there are:

$P(“good”,“good”,“nice”,“game”∣c)=P(“good”∣c)P(“good”∣c)P(“nice”∣S)P(“game”∣c) $

In the statistical calculation of $P("good"|c)$, the repeated words in each counted spam sample are counted multiple times.

$P("good"|c)=\frac{The total number of occurrences of "good" in each rating = c comment}{
The sum of the number of occurrences of all words (counting the number of repetitions) in each comment with ratings = c}$

#### 3.6.2 Bernoulli model

This more simplified method is to treat repeated words as if they only occur once.

$P(“good”,“good”,“nice”,“game”∣c)=P(“good”∣c)P(“nice”∣S)P(“game”∣c) $

Statistical calculation $P("Word"∣c)$:

$P("good"|c)=\frac{The number of rating = c comment which occurrences "good"}{
The sum of the number of occurrences of all words (Only count once) in each comment with ratings = c}$

Such a model is called a Bernoulli model (also called a binomial independent model). This way is more simplified and convenient. Of course, it loses the word frequency information, so the effect may be worse.

#### 3.6.3 Mixed model

This method does not consider the number of occurrences of repeated words when calculating the probability of a sentence, but considers the number of occurrences of repeated words when calculating the probability $P ("word" | c)$ of a word statistically. model.


```python
def SegmentLineToWordsSet(sentense):
    sentense = re.sub("[%s]+"%('"|#|$|%|&|\|(|)|\[|\]|*|+|\-|/|<|=|>|@|^|`|{|}|~|,|.|?|!|:|;'), ' ', sentense)
    sentense = re.sub("[%s]+"%('\''),'',sentense)
    #return set([x.lower() for x in re.split(r'[\s|,|;|.|/|\[|\]|;|\!|?|\'|\\|\)|\(|\"|@|&|#|-|=|*|%|>|<|^|-]\s*',sentense.strip()) if x and x not in stopSet and len(x) > 1])
    return set([x.lower() for x in re.split(r'[\s]\s*',sentense.strip()) if x])

def buildVocabularyList(dataset):
    dict_list = {}
    pattern = re.compile('[0-9]+')
    for row in dataset:
        words = list(SegmentLineToWordsList(str(row[0]))) #Words that appear multiple times in the same comment are counted only once
        #words = set()
        #words = words.union(SegmentLineToWordsSet(str(row[0])))
        for word in words:
            if word in stopSet or len(word) == 1:
            #if len(word) == 1:
                continue
            if pattern.findall(word):
                continue
            if word not in dict_list:
                dict_list[word] = [0,0,0,0,0,0] #0-10 is rating,11 is sum
            dict_list[word][row[1]] += 1
            dict_list[word][len(dict_list[word])-1] += 1
    for word in list(dict_list.keys()):
        if dict_list[word][len(dict_list[word])-1] < len(train) * 0.00002:
            del dict_list[word]
    return dict_list
train_dict = buildVocabularyList(train)
train_dict
```




    {'considering': [32, 89, 851, 1050, 2716, 4738],
     'cards': [1706, 4734, 43022, 52414, 128173, 230049],
     'main': [56, 174, 2028, 2778, 6738, 11774],
     'mechanic': [197, 580, 6260, 8386, 23817, 39240],
     'game': [15132, 31003, 282583, 373989, 1161466, 1864173],
     'surprisingly': [9, 43, 812, 2139, 7919, 10922],
     ...}



### Count the number of comments in each category to prepare for calculating the prior probability.


```python
def getRatingProbability(dataset):
    rating_num = [0,0,0,0,0]
    for row in dataset:
        rating_num[row[1]] += 1
    return rating_num
rating_num = getRatingProbability(dataset_org)
print(rating_num)
```

    [19158, 40146, 406379, 531411, 1534845]


### Count the number of words in each rating to prepare for the mixed model to calculate the conditional probability.


```python
def getClassWordNum(dataset):
    word_num = [0,0,0,0,0]
    for word in list(train_dict.keys()):
        for i in range(0,len(word_num)):
            word_num[i] += train_dict[word][i]
    return word_num
word_num = getClassWordNum(dataset_org)
print(word_num)
```

    [267947, 613791, 5883701, 7539791, 22245184]


### 3.7 Using smoothing

Smoothing techniques all give words that do not appear in the training set an estimated probability, and accordingly reduce the probability of other words that have already appeared. The smoothing technology is a real demand that arises because the data set is too small. If the data set is large enough, the effect of the smoothing technique on the results will become smaller. But because 1rating has a small number of comments, it makes sense to use smoothing techniques here.


For the Bernoulli model, a smoothing algorithm for $P("good"|c)$ is:

$P("good"|c)=\frac{The number of rating = c comment which occurrences "good" + lambda}{
The sum of the number of occurrences of all words (Only count once) in each comment with ratings = c + lambda * the number of ratings}$


For the Polynomial model, a smoothing algorithm for $P("good"|c)$ is:

$P("good"|c)=\frac{The total number of occurrences of "good" in each rating = c comment + lambda}{
The sum of the number of occurrences of all words (counting the number of repetitions) in each comment with ratings = c + lambda * The number of words in the vocabulary counted}$


$ 0<lambda<=1$


```python
lambda_value = 0.0005
lambda_cag = len(rating_num) * lambda_value
def getConditionalProbabilityUsingSmoothing(word):
    conditional_probability = list()
    for i in range(0,len(rating_num)):
        if word not in train_dict:
            pro = lambda_value/(len(train_dict)*lambda_value+word_num[i])
        else:
            pro = (lambda_value + train_dict[word][i])/(len(train_dict)*lambda_value+word_num[i])
        conditional_probability.append(pro)
    return conditional_probability

```

### predict rating function


```python
def predict(review):
    words = set()
    words = words.union(SegmentLineToWordsSet(review))
    probability = np.array(rating_num) / len(train)
    pattern = re.compile('[0-9]+')
    for word in words:
        if pattern.findall(word):
                continue
        if word not in stopSet and len(word) > 1:
        #if len(word) > 1:
            probability *= getConditionalProbabilityUsingSmoothing(word)
    probability = list(probability)
    return probability.index(max(probability))
```

### 4.Evaluation score

### 4.1 accuracy metric function


```python
def accuracy_metric(test_dataset):
    correct = 0
    for row in test_dataset:
        if row[1] == predict(str(row[0])):
            correct += 1
    return correct / float(len(test_dataset)) * 100.0
```

### 4.2 Using part of train set to get the evaluation score
Because the data set is very large and training takes a long time, so here we did not use the development set to verify, but use part of the training set to evaluate the model.


```python
train_part = list()
for i in range(1,1000):
    train_part.append(train[i])
print('Accuracy: %.3f%%' % accuracy_metric(train_part))
```

    Accuracy: 59.159%


### 4.3 Using part of test set to get the evaluation score


```python
test_part = list()
for i in range(1,10000):
    test_part.append(test[i])
print('Accuracy: %.3f%%' % accuracy_metric(test_part))
```

    Accuracy: 58.836%


### 4.4 Using all test set to get the evaluation score


```python
print('Accuracy: %.3f%%' % accuracy_metric(test))
```

    Accuracy: 59.153%


### 5.Save the trained model for future website development.


```python
f = open('dict_file.txt','w')
f.write(str(train_dict))
f.close()
```

### Load the saved model


```python
f = open('dict_file.txt','r')
a = f.read()
read_dictionary = eval(a)
f.close()
print(read_dictionary['greatest'])
print(len(read_dictionary))
print(len(train_dict))
```

    [21, 30, 369, 602, 2846, 3868]
    21242
    21242


### 6.Challenge

a.The entire data set is very large, and training the model once takes too long.

Solution:First, randomly sampled one-tenth of the data for algorithm design and model tuning. When the model optimization is completed, all data is used for training to obtain the final model.

b.There are special characters and other languages in the comments data, it is difficult to use a regular expression to match successfully.

Solution:By consulting multiple data and observing the comment text data for a long time, we finally use two regular expressions to match and observe whether the two results are the same. The first regular expression is to use ASCII, and the second regular expression is to replace punctuation with spaces. The symbol then uses spaces to cut the string.

c.The accuracy of the algorithm is very low, probably between 20% -30%.

Solution:Through many experiments and referred to the iPhone App Store and Amazon ’s 5-point rating standards, it is found that the score is mapped to the 1-5 interval, and then the discretization is performed, and the final algorithm accuracy rate is nearly doubled. And then by using Mixed model, smoothing and stop words, the final accuracy is improved to about 60%. But it is still not enough, if the reader has a better idea and don’t mind sharing with me, please email me, I will be grateful.


### 7.Hyper parameter tuning

a.smoothing lambda_value = 0.0005

b.Select Mixed model(Combined the Polynomial model and Bernoulli model) to posterior probability.

c.Map ratings to 1-5

### 8.References

Professor Mr. Park's data mining Naïve Bayes lecture.

https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

https://en.wikipedia.org/wiki/Naive_Bayes_classifier

https://gist.github.com/sebleier/554280

https://blog.csdn.net/longxinchen_ml/article/details/50597149
