---
title: Term Project
subtitle: Goal: Use the [board game geek review data](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews) to design an algorithm model that can predict the rating for the given  review.[Algorithm Git](https://github.com/VegetableCattle/youxinyu/tree/master/static/files/NBCproject.ipynb).[Download Algorithm file](https://yongli.netlify.com/files/NBCproject.ipynb). And then using this model to develop a predictor of reviews website. [Website Link](https://flask-rating-prediction.herokuapp.com/predicted). [Website Git](https://github.com/VegetableCattle/predictor-of-reviews)
summary: Goal: Use the [board game geek review data](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews) to design an algorithm model that can predict the rating for the given  review.[Algorithm Git](https://github.com/VegetableCattle/youxinyu/tree/master/static/files/NBCproject.ipynb).[Download Algorithm file](https://yongli.netlify.com/files/NBCproject.ipynb). And then using this model to develop a predictor of reviews website. [Website Link](https://flask-rating-prediction.herokuapp.com/predicted). [Website Github](https://github.com/VegetableCattle/predictor-of-reviews)
date: "2020-04-17T00:00:00Z"
comments: false
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""
  
---

## Goal: Use the [board game geek review data](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews) to design an algorithm model that can predict the rating for the given  review. And then using this model to develop a predictor of reviews website. [Website Link](https://flask-rating-prediction.herokuapp.com/predicted). [Website Github](https://github.com/VegetableCattle/predictor-of-reviews)

## Blog content: Introduction the implementation of Naive Bayse Classifier without using algorithm library.

### 1.Algorithm principle

### 1.1 Introduction

The Bayesian method is a method with a long history and a lot of theoretical foundations. At the same time, it is direct and efficient when dealing with many problems. Many advanced natural language processing models can also evolve from it. Therefore, learning the Bayesian method is a very good entry point for studying natural language processing problems.

### 1.2 Bayesian formula

####  Joint probability formula: $P(Y,X) = P(Y|X)P(X)=P(X|Y)P(Y) $

Among them, $P(Y) $ is called the prior probability, $P(Y∣X)$ is called the posterior probability, and $P(Y, X)$ is called the joint probability. This way we can derive the Bayesian formula.

#### Bayesian formula: $P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)} $

### 1.3 Naive Bayes Algorithm

The probability model classifier is a conditional probability model:

$p(C|F_1,…,F_n) $

The independent variable C has several categories and the conditions depend on several feature variables, but the problem is that if the dimension of the number of features n is large or each feature can take a large number of values, it is not realistic to list the probability table based on the probability model. So we modified this model to make it feasible. According to the Bayesian formula:

$p(C│F_1,…,F_n )=\frac{p(C)*p(F_1,…,F_n |C)}{p(F_1,…,F_n)} $

The denominator does not depend on C, and the value of the feature is also given, so the denominator can be considered a constant. The molecules are then equivalent to a joint distribution model:

$p(C|F_1,…,F_n) $

$∝p(C)*p(F_1,…,F_n│C) $

$∝p(C)*p(F_1│C)*p(F_2,…,F_n│C,F_1 ) $

$∝p(C)*p(F_1│C)*p(F_2│C,F_1 )p(F_3│C,F_1,F_2 )…p(F_n│C,F_1,F_2…F_(n-1) )$

Assuming that each feature is independent of other features, that is, the features are independent of each other, there is:

$p(F_i│C,F_j )=p(F_i│C) $

This means that the conditional distribution of the variable C can be expressed as:

$p(C│F_1,…,F_n )=\frac{1}{Z} p(C)*∏_i^np(F_i│C)$

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

Set the minimum threshold for the number of occurrences of words to eliminate the influence of rare words. After many experiments, the threshold value set as len (train) * 0.00002.

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

This method does not consider the number of occurrences of repeated words when calculating the probability of a sentence, but considers the number of occurrences of repeated words when calculating the probability P ("word" | c) of a word statistically. model.


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
     'poor': [292, 550, 3010, 2023, 3696, 9571],
     'quality': [258, 432, 3406, 4510, 16756, 25362],
     'graphic': [31, 86, 867, 1005, 2430, 4419],
     'design': [336, 745, 6285, 7146, 23152, 37664],
     'diamonds': [3, 5, 74, 140, 296, 518],
     'great': [449, 1151, 17835, 33006, 190373, 242814],
     'though': [296, 938, 14979, 22816, 53285, 92314],
     'look': [290, 532, 4291, 5313, 18379, 28805],
     'fun': [1237, 3949, 47032, 84474, 247949, 384641],
     'trump': [9, 19, 247, 323, 793, 1391],
     'trick': [38, 167, 2272, 3500, 7490, 13467],
     'taking': [142, 347, 3684, 5019, 12990, 22182],
     'players': [1069, 3068, 36989, 55985, 169995, 267106],
     'collect': [42, 137, 1451, 1922, 3982, 7534],
     'execute': [3, 24, 173, 206, 662, 1068],
     'special': [108, 359, 4917, 6409, 13481, 25274],
     'actions': [113, 409, 4822, 6562, 20284, 32190],
     'wonderfully': [1, 7, 115, 189, 1933, 2245],
     'easy': [170, 552, 9158, 18556, 74180, 102616],
     'play': [3211, 8693, 88052, 117345, 330361, 547662],
     'kids': [369, 1251, 14561, 15112, 23608, 54901],
     'choosing': [17, 51, 642, 884, 2488, 4082],
     'card': [1031, 2824, 25921, 33923, 84610, 148309],
     'mana': [3, 10, 103, 148, 519, 783],
     'zone': [12, 17, 169, 208, 555, 961],
     'always': [240, 572, 6715, 9939, 44179, 61645],
     'hard': [234, 650, 7378, 9968, 29996, 48226],
     'decision': [217, 407, 2786, 2757, 9043, 15210],
     'nice': [248, 985, 14382, 29406, 77103, 122124],
     'ccg': [40, 111, 982, 982, 3184, 5299],
     'extra': [75, 205, 2137, 3142, 10084, 15643],
     'stuff': [156, 322, 3293, 3789, 10066, 17626],
     'coins': [19, 70, 490, 750, 2309, 3638],
     'counters': [68, 174, 1453, 1648, 5463, 8806],
     'etc': [102, 321, 3132, 3720, 10505, 17780],
     'needed': [94, 217, 2153, 2664, 6800, 11928],
     'shields': [1, 11, 111, 116, 364, 603],
     'provide': [35, 100, 873, 1216, 3904, 6128],
     'means': [149, 306, 2858, 3079, 8453, 14845],
     'see': [594, 1552, 14538, 17246, 43185, 77115],
     'winning': [186, 374, 2912, 3481, 9494, 16447],
     'production': [115, 289, 3098, 3841, 11273, 18616],
     'deluxe': [14, 26, 272, 507, 2969, 3788],
     'version': [347, 795, 9359, 13708, 41289, 65498],
     'insane': [25, 35, 218, 227, 784, 1289],
     'box': [393, 639, 5440, 7630, 25627, 39729],
     'super': [83, 155, 1959, 2843, 11638, 16678],
     'heavy': [57, 184, 2080, 3140, 11843, 17304],
     'little': [477, 1546, 22587, 37696, 82248, 144554],
     'bigger': [26, 63, 832, 1259, 3751, 5931],
     'money': [544, 728, 4993, 6078, 17034, 29377],
     'really': [1298, 4015, 43180, 49539, 163686, 261718],
     'tight': [9, 39, 769, 1615, 8298, 10730],
     'one': [2390, 5737, 54370, 69448, 219112, 351057],
     'race': [65, 224, 3416, 5243, 14085, 23033],
     'aspect': [68, 234, 2839, 3863, 11216, 18220],
     'regarding': [18, 26, 272, 401, 1197, 1914],
     'goals': [36, 79, 890, 1250, 4097, 6352],
     'issue': [63, 110, 1682, 2245, 5888, 9988],
     'variability': [2, 14, 315, 648, 3237, 4216],
     'thing': [451, 1201, 9852, 11506, 30040, 53050],
     'feel': [341, 920, 12233, 17061, 47725, 78280],
     'right': [353, 805, 9363, 13720, 35446, 59687],
     'threat': [8, 22, 249, 246, 1039, 1564],
     'excitement': [33, 105, 978, 903, 2115, 4134],
     'level': [135, 327, 3236, 4594, 15994, 24286],
     'dinosaurs': [6, 18, 161, 223, 476, 884],
     'tier': [5, 14, 135, 196, 618, 968],
     'bit': [189, 667, 17208, 33614, 74489, 126167],
     'strange': [27, 108, 1132, 1345, 2287, 4899],
     'plays': [365, 1025, 16125, 28635, 93843, 139993],
     'smoothly': [5, 18, 234, 518, 2456, 3231],
     'interesting': [385, 1447, 22250, 31548, 68862, 124492],
     'mechanics': [310, 981, 9927, 13144, 44903, 69265],
     'feels': [186, 629, 9020, 10999, 25715, 46549],
     'like': [2218, 6198, 68314, 85119, 223449, 385298],
     'lesson': [20, 34, 175, 159, 478, 866],
     'near': [45, 141, 1224, 1396, 4008, 6814],
     'corporate': [4, 4, 40, 37, 214, 299],
     'capital': [9, 12, 128, 142, 574, 865],
     'investment': [14, 40, 574, 657, 2054, 3339],
     'clearly': [125, 199, 1287, 1224, 3090, 5925],
     'shows': [41, 88, 692, 851, 2097, 3769],
     'upgrades': [5, 29, 295, 357, 1493, 2179],
     'investments': [2, 1, 55, 85, 307, 450],
     'company': [215, 155, 838, 1019, 3498, 5725],
     'fail': [72, 126, 633, 531, 1334, 2696],
     'short': [89, 355, 4860, 6933, 19913, 32150],
     'term': [23, 78, 967, 1399, 4699, 7166],
     'return': [28, 51, 516, 607, 1503, 2705],
     'well': [576, 1568, 18413, 29087, 106205, 155849],
     'designed': [172, 296, 2069, 2430, 8617, 13584],
     'strategies': [31, 105, 1657, 3025, 15925, 20743],
     'costly': [3, 12, 78, 103, 356, 552],
     'simulation': [58, 93, 803, 979, 3713, 5646],
     'maximize': [2, 16, 198, 376, 1304, 1896],
     'turnover': [0, 0, 16, 14, 52, 82],
     'purchased': [60, 70, 699, 1059, 4203, 6091],
     'friend': [85, 169, 1255, 1584, 5310, 8403],
     'possession': [2, 4, 37, 42, 146, 231],
     'rules': [1268, 2837, 24354, 29186, 93617, 151262],
     'spanish': [16, 17, 163, 268, 885, 1349],
     'impressions': [8, 33, 387, 509, 1471, 2408],
     'filled': [16, 58, 396, 500, 1750, 2720],
     'decisions': [388, 941, 7562, 7889, 27699, 44479],
     'immediate': [5, 20, 174, 237, 913, 1349],
     'versus': [10, 23, 283, 458, 1422, 2196],
     'delayed': [5, 4, 45, 48, 166, 268],
     'payoffs': [0, 3, 34, 27, 82, 146],
     'plenty': [28, 96, 1011, 1729, 8985, 11849],
     'strategic': [67, 226, 2683, 4388, 19667, 27031],
     'options': [87, 285, 3403, 4590, 16479, 24844],
     'presentation': [8, 44, 531, 623, 1868, 3074],
     'advice': [18, 15, 105, 110, 555, 803],
     'take': [502, 1418, 12424, 16117, 42446, 72907],
     'starting': [79, 176, 1643, 2442, 7226, 11566],
     'customer': [53, 23, 72, 102, 407, 657],
     'training': [11, 15, 123, 176, 622, 947],
     'wheels': [5, 9, 107, 135, 443, 699],
     'soon': [80, 154, 1472, 1744, 6932, 10382],
     'possible': [220, 345, 3304, 4091, 12090, 20050],
     'eliminate': [12, 35, 249, 253, 617, 1166],
     'auto': [16, 20, 159, 185, 412, 792],
     'pilot': [4, 15, 102, 158, 390, 669],
     'making': [314, 655, 5233, 5930, 16981, 29113],
     'first': [879, 2129, 21505, 28574, 88886, 141973],
     'turns': [229, 830, 6702, 7304, 19324, 34389],
     'pleasant': [6, 26, 579, 1540, 3074, 5225],
     'enough': [435, 1196, 15952, 21466, 50486, 89535],
     'think': [575, 1550, 18810, 27392, 69482, 117809],
     'suffers': [25, 104, 1339, 1461, 1984, 4913],
     'badly': [93, 165, 974, 748, 1661, 3641],
     'comparison': [28, 44, 650, 779, 1735, 3236],
     'puerto': [22, 56, 657, 841, 4231, 5807],
     'rico': [21, 55, 649, 838, 4215, 5778],
     'compared': [44, 134, 1718, 2079, 5160, 9135],
     'draw': [327, 888, 7508, 7331, 14164, 30218],
     'influence': [37, 182, 1401, 1787, 4978, 8385],
     'success': [30, 77, 598, 817, 2660, 4182],
     'much': [1255, 4060, 48912, 49584, 111053, 214864],
     'endgame': [19, 70, 944, 991, 1999, 4023],
     'still': [544, 1164, 15947, 26751, 79719, 124125],
     'lack': [121, 259, 2591, 2751, 5579, 11301],
     'arduous': [1, 5, 29, 19, 52, 106],
     'setup': [56, 188, 2011, 3084, 11074, 16413],
     'certainly': [48, 159, 2546, 3956, 8838, 15547],
     'makes': [450, 915, 8773, 11911, 40558, 62607],
     'appealing': [18, 63, 754, 883, 2252, 3970],
     'nothing': [636, 1267, 10952, 9361, 12052, 34268],
     'wrong': [256, 441, 3702, 3672, 8000, 16071],
     'similar': [99, 300, 4333, 6259, 15093, 26084],
     'eminent': [1, 3, 58, 78, 184, 324],
     'domain': [6, 18, 174, 194, 371, 763],
     'artificially': [4, 8, 82, 55, 83, 232],
     'balanced': [53, 140, 2104, 3086, 14204, 19587],
     'score': [159, 381, 4523, 7225, 18160, 30448],
     'wise': [17, 40, 524, 683, 1913, 3177],
     'everything': [253, 562, 3673, 3990, 16697, 25175],
     'happens': [104, 262, 1550, 1443, 3084, 6443],
     'blends': [0, 5, 32, 63, 467, 567],
     'together': [164, 309, 3199, 3968, 15018, 22658],
     'bland': [49, 143, 1957, 1307, 1329, 4785],
     'paste': [13, 11, 112, 83, 264, 483],
     'unoriginal': [10, 16, 88, 46, 50, 210],
     'meh': [54, 157, 3539, 1820, 936, 6506],
     'thrift': [29, 69, 621, 761, 747, 2227],
     'store': [71, 122, 892, 1128, 2474, 4687],
     'find': [314, 907, 9442, 11271, 29002, 50936],
     'pillars': [3, 3, 133, 218, 711, 1068],
     'earth': [29, 64, 477, 576, 1870, 3016],
     'themed': [45, 132, 1359, 1854, 5129, 8519],
     'games': [2347, 5070, 47635, 55065, 176222, 286339],
     'clunky': [37, 115, 1177, 967, 1300, 3596],
     'flipping': [36, 56, 425, 433, 816, 1766],
     'luck': [622, 1989, 19163, 23110, 49421, 94305],
     'based': [303, 785, 8027, 9616, 23008, 41739],
     'theme': [620, 1784, 20271, 27128, 79139, 128942],
     'co': [69, 192, 2593, 4259, 15999, 23112],
     'op': [53, 146, 1917, 3364, 13132, 18612],
     'artwork': [159, 482, 4549, 6271, 20132, 31593],
     'way': [906, 2380, 19819, 19358, 50304, 92767],
     'cool': [144, 486, 5430, 7208, 20571, 33839],
     'enjoyed': [61, 195, 3330, 7328, 26393, 37307],
     'usually': [108, 315, 3804, 5186, 14680, 24093],
     'regular': [41, 109, 1494, 1914, 5173, 8731],
     'decks': [64, 154, 1632, 2117, 7361, 11328],
     'ok': [149, 445, 9272, 11932, 8230, 30028],
     'afternoon': [7, 19, 116, 179, 708, 1029],
     'among': [28, 73, 745, 1002, 3611, 5459],
     'several': [129, 321, 3704, 4910, 15602, 24666],
     'generations': [4, 3, 24, 55, 210, 296],
     'wifes': [3, 11, 206, 305, 1356, 1881],
     'grandmother': [3, 6, 72, 62, 158, 301],
     'cheats': [1, 2, 17, 17, 46, 83],
     'non': [180, 373, 3966, 6098, 18047, 28664],
     'stop': [142, 289, 1694, 1787, 4973, 8885],
     'trying': [280, 701, 6173, 7943, 21439, 36536],
     'reserve': [6, 15, 110, 212, 609, 952],
     'piles': [10, 36, 307, 382, 821, 1556],
     'turn': [573, 1708, 14324, 16651, 44087, 77343],
     'get': [1479, 3521, 32559, 41307, 119873, 198739],
     'people': [1019, 1915, 14916, 19314, 57584, 94748],
     'extremely': [136, 546, 2464, 2257, 8348, 13751],
     'annoying': [126, 588, 2315, 1834, 2830, 7693],
     'ever': [1307, 1679, 6761, 5748, 26793, 42288],
     'played': [1753, 4082, 41158, 53471, 158441, 258905],
     'couple': [137, 346, 4552, 5893, 15543, 26471],
     'times': [324, 946, 9626, 12790, 36491, 60177],
     'felt': [164, 542, 7207, 7063, 9355, 24331],
     'long': [615, 2197, 20476, 22653, 50254, 96195],
     'cumbersome': [14, 25, 359, 369, 649, 1416],
     'blind': [50, 163, 1503, 1607, 2873, 6196],
     'bidding': [27, 184, 2683, 4150, 10932, 17976],
     'spells': [11, 35, 392, 496, 1371, 2305],
     'seem': [155, 388, 6528, 7760, 16265, 31096],
     'balance': [127, 197, 2352, 3441, 13664, 19781],
     'awesome': [51, 113, 1352, 1945, 18223, 21684],
     'party': [196, 475, 5896, 8182, 19674, 34423],
     'best': [346, 787, 9596, 14989, 80782, 106500],
     'max': [24, 43, 466, 738, 1976, 3247],
     'player': [1013, 2831, 29960, 41920, 133767, 209491],
     'count': [81, 169, 1424, 1955, 6331, 9960],
     'longest': [7, 20, 153, 220, 658, 1058],
     'time': [1408, 3433, 30894, 39632, 125276, 200643],
     'thought': [250, 637, 5245, 6137, 16290, 28559],
     'uninteresting': [39, 166, 1116, 375, 299, 1995],
     'eventually': [49, 104, 956, 1046, 2682, 4837],
     'realised': [10, 19, 118, 122, 344, 613],
     'things': [321, 803, 7811, 9527, 27155, 45617],
     'going': [350, 924, 8864, 10651, 28819, 49608],
     'react': [7, 14, 163, 217, 697, 1098],
     'follow': [32, 77, 782, 1076, 3316, 5283],
     'although': [76, 258, 3885, 6670, 18867, 29756],
     'listed': [20, 33, 240, 344, 835, 1472],
     'cannot': [182, 313, 1783, 1785, 5205, 9268],
     'agree': [36, 76, 600, 695, 1937, 3344],
     'terrific': [2, 4, 109, 278, 2280, 2673],
     'action': [147, 508, 5305, 7440, 22613, 36013],
     'selection': [31, 99, 1350, 2222, 6615, 10317],
     'splendid': [22, 1, 9, 39, 299, 370],
     'mech': [2, 5, 74, 114, 355, 550],
     'minis': [84, 137, 1239, 1426, 5477, 8363],
     'invoke': [1, 5, 28, 34, 85, 153],
     'combative': [1, 0, 20, 31, 126, 178],
     'atmosphere': [16, 27, 421, 629, 2694, 3787],
     'realized': [52, 103, 641, 554, 1395, 2745],
     'however': [215, 528, 8448, 12027, 24975, 46193],
     'serve': [18, 42, 259, 261, 710, 1290],
     'solidly': [0, 1, 67, 99, 259, 426],
     'useful': [44, 116, 950, 901, 2447, 4458],
     'purpose': [49, 88, 543, 550, 1208, 2438],
     'beyond': [132, 175, 1252, 1062, 2657, 5278],
     'aggressive': [10, 34, 316, 516, 1839, 2715],
     'image': [36, 19, 126, 155, 490, 826],
     'impose': [3, 1, 11, 18, 64, 97],
     'mechanisms': [44, 158, 1994, 2665, 7861, 12722],
     'whole': [227, 448, 4166, 4265, 12104, 21210],
     'package': [22, 40, 454, 735, 3226, 4477],
     'comes': [130, 294, 3223, 4184, 12572, 20403],
     'beautifully': [6, 15, 243, 361, 2434, 3059],
     'made': [486, 900, 6690, 7383, 22997, 38456],
     'playing': [1139, 2702, 24932, 30331, 92575, 151679],
     'mediocre': [46, 105, 1497, 687, 585, 2920],
     'overall': [79, 236, 4033, 5804, 15203, 25355],
     'plus': [96, 289, 2477, 3384, 11445, 17691],
     'side': [95, 276, 2982, 4032, 11532, 18917],
     'technology': [3, 27, 265, 340, 1268, 1903],
     'generally': [39, 133, 1695, 2064, 4770, 8701],
     'fairly': [35, 171, 3890, 6164, 12948, 23208],
     'cost': [101, 122, 1229, 1645, 4853, 7950],
     'everyone': [277, 669, 5996, 7756, 26241, 40939],
     'also': [528, 1368, 16886, 23145, 72310, 114237],
     'liked': [89, 300, 4256, 6547, 16066, 27258],
     'dual': [1, 6, 92, 117, 471, 687],
     'downside': [2, 18, 323, 808, 4079, 5230],
     'else': [302, 693, 5926, 5328, 11119, 23368],
     'theres': [371, 967, 10478, 11436, 26331, 49583],
     'ton': [21, 84, 830, 1217, 4946, 7098],
     'massive': [45, 70, 627, 559, 1979, 3280],
     'amount': [114, 344, 3454, 5255, 17473, 26640],
     'worth': [223, 523, 4925, 6270, 16967, 28908],
     'even': [1625, 3096, 17862, 17109, 61563, 101255],
     'ignored': [17, 21, 138, 149, 426, 751],
     'face': [71, 164, 1524, 2266, 5987, 10012],
     'ship': [65, 173, 1559, 1901, 5899, 9597],
     'movement': [94, 275, 2991, 3874, 9696, 16930],
     'plotting': [4, 10, 63, 64, 272, 413],
     'requires': [66, 145, 1604, 2333, 7600, 11748],
     'trust': [19, 24, 132, 174, 712, 1061],
     'least': [438, 903, 7333, 7496, 16254, 32424],
     'minutes': [256, 692, 4579, 5369, 16748, 27644],
     'playtime': [9, 55, 498, 657, 2462, 3681],
     'vastly': [20, 31, 290, 271, 742, 1354],
     'overstayed': [2, 6, 83, 55, 47, 193],
     'welcome': [23, 106, 1138, 1648, 3429, 6344],
     'includes': [23, 44, 547, 1028, 4943, 6585],
     'many': [744, 1870, 18181, 20094, 61596, 102485],
     'set': [269, 740, 8360, 12963, 38834, 61166],
     'collection': [146, 311, 4012, 6735, 19912, 31116],
     'ive': [622, 1080, 8908, 11949, 47358, 69917],
     'yet': [193, 421, 4460, 7400, 28685, 41159],
     'enjoy': [265, 808, 9291, 13176, 46112, 69652],
     'morels': [0, 0, 16, 21, 93, 130],
     'offers': [24, 111, 1196, 1798, 6433, 9562],
     'tough': [23, 44, 1255, 2518, 10797, 14637],
     'pacing': [2, 13, 204, 256, 690, 1165],
     'entering': [7, 10, 71, 86, 241, 415],
     'wandering': [5, 11, 116, 89, 172, 393],
     'row': [39, 85, 1070, 1622, 4116, 6932],
     'shifting': [1, 20, 168, 273, 845, 1307],
     'decay': [0, 4, 10, 28, 62, 104],
     'pile': [100, 170, 1178, 1298, 2333, 5079],
     'brilliant': [17, 44, 531, 883, 9841, 11316],
     'circular': [1, 4, 72, 81, 153, 311],
     'fiddly': [88, 367, 3863, 3975, 7296, 15589],
     'grand': [14, 24, 310, 471, 2212, 3031],
     'scoring': [87, 404, 5662, 8638, 21916, 36707],
     'rich': [25, 83, 635, 674, 2970, 4387],
     'art': [252, 675, 5938, 7229, 22607, 36701],
     'fabulous': [4, 7, 72, 139, 1047, 1269],
     'quick': [68, 243, 6791, 15900, 47443, 70445],
     'warrant': [5, 16, 283, 332, 568, 1204],
     'another': [464, 1186, 10955, 12082, 30527, 55214],
     'every': [460, 1043, 7883, 9336, 39543, 58265],
     'hat': [6, 18, 156, 163, 510, 853],
     'gameplay': [347, 871, 8160, 8718, 27831, 45927],
     'definitely': [81, 271, 3933, 7651, 24845, 36781],
     'mood': [12, 45, 1080, 2528, 2982, 6647],
     'cut': [44, 91, 929, 1253, 3878, 6195],
     'thrust': [3, 5, 33, 40, 114, 195],
     'style': [67, 276, 3440, 4754, 13010, 21547],
     'light': [113, 414, 9130, 18656, 37878, 66191],
     'gateway': [10, 58, 1228, 2887, 10987, 15170],
     'heroscape': [5, 18, 159, 231, 723, 1136],
     'wonderful': [23, 49, 668, 1134, 10270, 12144],
     'family': [122, 330, 5866, 10884, 32234, 49436],
     'brings': [22, 48, 546, 810, 3232, 4658],
     'minatures': [4, 7, 37, 46, 157, 251],
     'along': [85, 218, 2107, 2790, 8761, 13961],
     'typically': [14, 42, 466, 608, 1704, 2834],
     'last': [283, 703, 5793, 6887, 18991, 32657],
     'hour': [135, 419, 2494, 2774, 9057, 14879],
     'half': [195, 476, 3631, 3618, 8404, 16324],
     'hours': [350, 829, 4060, 3932, 15313, 24484],
     'bought': [233, 439, 3876, 4547, 14977, 24072],
     'master': [17, 50, 664, 1077, 6284, 8092],
     'bucks': [20, 37, 211, 243, 537, 1048],
     'bringing': [11, 21, 212, 298, 1153, 1695],
     'total': [149, 204, 1262, 1483, 4267, 7365],
     'ms': [1, 1, 21, 26, 77, 126],
     'two': [434, 1448, 14888, 21140, 66407, 104317],
     'kinda': [54, 120, 1689, 1880, 2720, 6463],
     'stupid': [292, 424, 1451, 930, 1757, 4854],
     'needs': [115, 266, 3585, 5144, 11946, 21056],
     'house': [106, 284, 2867, 3250, 8366, 14873],
     'make': [1055, 2107, 18225, 21300, 61089, 103776],
     'fair': [51, 112, 1655, 2142, 4878, 8838],
     'would': [1331, 2833, 28970, 35618, 75828, 144580],
     'real': [344, 772, 6364, 6571, 18980, 33031],
     'involved': [105, 254, 2682, 3655, 10578, 17274],
     'figures': [38, 77, 903, 1014, 2906, 4938],
     'hand': [182, 648, 6527, 8204, 19949, 35510],
     'painted': [13, 27, 321, 500, 2516, 3377],
     'professional': [11, 17, 76, 80, 286, 470],
     'miniatures': [118, 162, 1567, 1854, 7680, 11381],
     'painter': [0, 2, 11, 15, 64, 92],
     'uk': [14, 32, 254, 289, 1083, 1672],
     'bad': [1095, 2307, 16558, 17914, 22660, 60534],
     'offer': [39, 132, 1488, 1639, 4144, 7442],
     'want': [627, 1392, 12287, 15016, 43275, 72597],
     'teach': [88, 163, 2062, 4348, 21254, 27915],
     'gamers': [105, 253, 4164, 7495, 26078, 38095],
     'german': [47, 84, 1150, 1748, 6426, 9455],
     'settlers': [22, 89, 973, 1440, 3992, 6516],
     'catan': [24, 74, 1043, 1576, 4300, 7017],
     'racked': [0, 0, 2, 10, 41, 53],
     'lot': [360, 1182, 17842, 26523, 89008, 134915],
     'years': [274, 461, 4295, 4949, 16446, 26425],
     'ago': [82, 204, 1691, 1637, 3717, 7331],
     'fond': [19, 50, 891, 981, 1675, 3616],
     'anymore': [75, 148, 1403, 1498, 2710, 5834],
     'used': [196, 466, 4976, 6581, 17080, 29299],
     'dice': [885, 2303, 19886, 23522, 53068, 99664],
     'affect': [14, 43, 507, 624, 1933, 3121],
     'pretty': [320, 1158, 14979, 21390, 41331, 79178],
     'negotiation': [10, 116, 1192, 1363, 3982, 6663],
     'trading': [37, 148, 1633, 2383, 6241, 10442],
     'phase': [84, 175, 1640, 2005, 6140, 10044],
     'often': [113, 375, 5437, 8032, 21253, 35210],
     'drags': [44, 124, 974, 758, 795, 2695],
     'unnecessarily': [11, 37, 293, 244, 364, 949],
     'introductory': [5, 18, 356, 590, 1627, 2596],
     'new': [324, 615, 7801, 12556, 51079, 72375],
     'lots': [122, 406, 5234, 8367, 39378, 53507],
     'better': [870, 2228, 26806, 32049, 73996, 135949],
     'nowadays': [7, 51, 376, 342, 629, 1405],
     'different': [286, 717, 9553, 16059, 63339, 89954],
     'big': [222, 553, 6815, 8722, 24311, 40623],
     'fits': [9, 52, 625, 1115, 4231, 6032],
     'help': [119, 361, 3097, 3842, 10348, 17767],
     'lighthearted': [2, 4, 87, 203, 449, 745],
     'enjoyable': [77, 235, 3307, 7864, 24488, 35971],
     'filler': [66, 221, 6793, 16028, 27448, 50556],
     'wind': [16, 24, 276, 355, 918, 1589],
     'opinion': [123, 213, 2064, 2652, 7436, 12488],
     'spinning': [11, 23, 124, 131, 262, 551],
     'wheel': [24, 49, 290, 407, 1117, 1887],
     'three': [220, 534, 5698, 7463, 22284, 36199],
     'points': [255, 834, 9996, 14327, 36051, 61463],
     'idea': [348, 797, 7683, 6962, 11270, 27060],
     'rule': [212, 471, 3961, 4824, 14426, 23894],
     'actually': [501, 1055, 7973, 9497, 23485, 42511],
     'quite': [195, 685, 11085, 19538, 50563, 82066],
     'clever': [63, 233, 3334, 5335, 17041, 26006],
     'add': [156, 409, 4048, 5385, 18518, 28516],
     'edit': [94, 129, 1198, 1798, 6492, 9711],
     'higher': [87, 184, 2381, 4323, 10689, 17664],
     'deal': [47, 138, 1455, 2054, 6057, 9751],
     'tiles': [157, 457, 6000, 9025, 24255, 39894],
     'delicious': [2, 3, 49, 91, 702, 847],
     'mouth': [11, 24, 185, 149, 273, 642],
     'watering': [1, 0, 13, 28, 64, 106],
     'weve': [28, 66, 660, 1080, 4498, 6332],
     'managed': [26, 92, 594, 735, 2846, 4293],
     'pieces': [241, 556, 5652, 7033, 17787, 31269],
     'atop': [1, 0, 17, 15, 73, 106],
     'others': [190, 510, 5711, 7631, 19139, 33181],
     'without': [412, 852, 7227, 8415, 26337, 43243],
     'knocking': [1, 8, 127, 145, 383, 664],
     'either': [239, 627, 6205, 6299, 12107, 25477],
     'good': [1094, 2922, 37700, 65191, 186114, 293021],
     'looking': [201, 452, 4338, 5844, 22494, 33329],
     'pics': [0, 4, 29, 46, 120, 199],
     'geek': [34, 61, 470, 601, 2045, 3211],
     'probably': [308, 763, 9995, 13221, 29751, 54038],
     'missing': [108, 218, 1989, 2293, 3742, 8350],
     'copy': [209, 301, 3056, 4520, 15679, 23765],
     'case': [104, 220, 1926, 2082, 5467, 9799],
     'variety': [31, 104, 1733, 2997, 13455, 18320],
     'shapes': [6, 16, 264, 355, 717, 1358],
     'try': [348, 875, 10301, 13492, 31461, 56477],
     'onto': [24, 84, 815, 1026, 2320, 4269],
     'cylinders': [2, 4, 35, 41, 163, 245],
     'keep': [226, 529, 5856, 8456, 25963, 41030],
     'standing': [13, 29, 261, 325, 804, 1432],
     'url': [45, 327, 960, 1835, 5905, 9072],
     'https': [64, 116, 445, 905, 4506, 6036],
     'www': [85, 118, 923, 1625, 6293, 9044],
     'boardgamegeek': [103, 120, 755, 1437, 5497, 7912],
     'com': [141, 270, 1633, 2850, 11647, 16541],
     'geeklist': [20, 25, 236, 545, 1233, 2059],
     'item': [34, 64, 575, 797, 1663, 3133],
     'uwe': [4, 12, 122, 191, 1358, 1687],
     'rosenberg': [8, 6, 108, 157, 977, 1256],
     'kind': [240, 669, 8186, 8795, 15220, 33110],
     'borrowing': [0, 2, 27, 38, 113, 180],
     'bunch': [87, 187, 1578, 1562, 3894, 7308],
     'heavier': [4, 15, 331, 738, 2443, 3531],
     'worker': [69, 179, 2721, 4698, 22428, 30095],
     'placement': [63, 255, 3931, 6752, 27773, 38774],
     'buildings': [42, 103, 1841, 3067, 9185, 14238],
     'unique': [41, 137, 2191, 4468, 19236, 26073],
     'powers': [51, 163, 2169, 3172, 9729, 15284],
     'forests': [0, 2, 28, 49, 167, 246],
     'clear': [90, 161, 1895, 2294, 7121, 11561],
     'ships': [73, 200, 1656, 2058, 5944, 9931],
     'boards': [34, 89, 891, 1418, 5596, 8028],
     'cover': [41, 79, 651, 766, 2244, 3781],
     'negative': [79, 112, 983, 1316, 4024, 6514],
     'vp': [33, 101, 1350, 2128, 4463, 8075],
     'streamlined': [15, 51, 759, 1317, 5205, 7347],
     'learned': [31, 70, 552, 707, 2920, 4280],
     'faster': [32, 112, 1188, 1880, 5252, 8464],
     'throughout': [21, 41, 578, 804, 3827, 5271],
     'might': [418, 978, 9518, 10716, 24887, 46517],
     'childhood': [21, 92, 992, 764, 1050, 2919],
     'memories': [32, 82, 1054, 1086, 2168, 4422],
     'love': [283, 814, 9092, 14154, 95075, 119418],
     'ios': [6, 23, 356, 744, 2598, 3727],
     'standard': [69, 122, 1832, 2358, 6236, 10617],
     'engine': [48, 127, 1858, 2709, 9430, 14172],
     'wp': [7, 8, 82, 195, 560, 852],
     'gathering': [31, 51, 549, 794, 2682, 4107],
     'resources': [64, 236, 2671, 3954, 11979, 18904],
     'convert': [8, 19, 220, 315, 681, 1243],
     'exciting': [42, 144, 2677, 2927, 8105, 13895],
     'worried': [7, 9, 127, 250, 1192, 1585],
     'relying': [3, 16, 63, 92, 215, 389],
     'looked': [51, 132, 940, 731, 1781, 3635],
     'past': [56, 131, 1152, 1292, 3718, 6349],
     'happy': [59, 109, 954, 1754, 6068, 8944],
     'proven': [5, 8, 68, 91, 401, 573],
     'convoluted': [35, 119, 718, 595, 871, 2338],
     'complicated': [73, 290, 2413, 2630, 7215, 12621],
     'fit': [55, 131, 1360, 1744, 4997, 8287],
     'back': [414, 830, 6806, 8206, 23429, 39685],
     'gave': [166, 377, 2210, 1841, 4020, 8614],
     'preferred': [4, 20, 414, 643, 1601, 2682],
     'less': [297, 726, 8703, 11258, 27846, 48830],
     'meant': [45, 108, 799, 705, 1399, 3056],
     'expansion': [116, 226, 3045, 6196, 32583, 42166],
     'could': [855, 1901, 15762, 16857, 39823, 75198],
     'workers': [17, 68, 983, 1791, 5782, 8641],
     'precious': [8, 18, 134, 140, 507, 807],
     'need': [394, 877, 10519, 16805, 46324, 74919],
     'come': [231, 488, 4940, 5907, 18140, 29706],
     'cant': [543, 1302, 9213, 9225, 28022, 48305],
     'ignore': [20, 42, 363, 393, 1199, 2017],
     'figured': [24, 48, 446, 555, 1499, 2572],
     'rush': [9, 28, 414, 497, 1234, 2182],
     'end': [479, 1303, 12801, 15723, 37001, 67307],
     'producing': [13, 15, 144, 170, 600, 942],
     'cheap': [117, 191, 1514, 1798, 4074, 7694],
     'attractions': [0, 2, 23, 62, 111, 198],
     'early': [103, 318, 3628, 4344, 10640, 19033],
     'economy': [6, 31, 373, 498, 2205, 3113],
     'restaurants': [0, 0, 27, 45, 154, 226],
     'hurt': [19, 61, 601, 874, 2316, 3871],
     'able': [160, 364, 3092, 4083, 12485, 20184],
     'visitors': [2, 1, 34, 81, 257, 375],
     'harder': [25, 64, 901, 1498, 4640, 7128],
     'buy': [445, 592, 4399, 5449, 17042, 27927],
     'arch': [0, 9, 32, 43, 78, 162],
     'ended': [88, 238, 1699, 1857, 3875, 7757],
     'win': [487, 1011, 8513, 10148, 32319, 52478],
     'triggered': [4, 20, 116, 167, 403, 710],
     'endings': [1, 2, 36, 66, 217, 322],
     'remarkable': [4, 3, 95, 107, 548, 757],
     'highly': [53, 197, 1546, 2118, 14353, 18267],
     'variable': [12, 80, 721, 1300, 5249, 7362],
     'triggers': [5, 8, 81, 135, 395, 624],
     'chosen': [13, 32, 350, 447, 1331, 2173],
     'pool': [21, 57, 450, 732, 1946, 3206],
     'length': [32, 152, 1940, 2806, 8579, 13509],
     'drafting': [30, 102, 1779, 3012, 10158, 15081],
     'research': [24, 58, 304, 355, 1274, 2015],
     'portion': [14, 24, 264, 326, 862, 1490],
     'favourite': [17, 24, 569, 1038, 9171, 10819],
     'work': [402, 986, 8805, 8644, 22797, 41634],
     'partly': [1, 13, 184, 251, 599, 1048],
     'use': [303, 616, 6511, 9259, 28887, 45576],
     'differing': [1, 2, 77, 127, 476, 683],
     'strength': [18, 55, 608, 842, 2357, 3880],
     'choose': [90, 237, 2511, 3419, 9309, 15566],
     'completely': [375, 725, 3901, 2945, 7824, 15770],
     'screwed': [38, 83, 722, 614, 1273, 2730],
     'unfocused': [0, 1, 30, 24, 27, 82],
     'bloated': [15, 38, 183, 150, 251, 637],
     'mix': [20, 92, 1250, 2089, 8776, 12227],
     'feed': [10, 15, 137, 228, 945, 1335],
     'nicely': [19, 41, 832, 1860, 6629, 9381],
     'resource': [54, 147, 1773, 2825, 10197, 14996],
     'aspects': [28, 89, 1066, 1344, 4402, 6929],
     'invest': [18, 42, 458, 544, 1479, 2541],
     'upgrade': [12, 53, 481, 714, 2924, 4184],
     'spots': [15, 36, 528, 728, 1914, 3221],
     'tied': [14, 27, 261, 359, 1186, 1847],
     'section': [20, 26, 294, 427, 1310, 2077],
     'unlike': [27, 64, 807, 1097, 4173, 6168],
     'parts': [93, 188, 1781, 2041, 5305, 9408],
     'solitaire': [66, 241, 2501, 3046, 9280, 15134],
     'sold': [167, 391, 4119, 4739, 7080, 16496],
     'interaction': [160, 438, 4522, 6000, 20399, 31519],
     'phases': [17, 41, 397, 549, 1975, 2979],
     'breather': [0, 0, 3, 7, 38, 48],
     'btw': [16, 14, 91, 116, 412, 649],
     'tried': [232, 502, 3320, 2838, 7629, 14521],
     'medium': [48, 27, 534, 1299, 6262, 8170],
     'warps': [2, 0, 10, 8, 38, 58],
     'racing': [20, 77, 1353, 2396, 6199, 10045],
     'objectives': [19, 52, 625, 824, 3096, 4616],
     'sure': [288, 716, 8840, 12801, 28062, 50707],
     'fast': [97, 291, 5189, 10593, 37939, 54109],
     'rather': [430, 1145, 12150, 11814, 18611, 44150],
     'building': [169, 475, 6437, 10086, 36876, 54043],
     'prefer': [71, 234, 4761, 7359, 15366, 27791],
     'longer': [113, 304, 3042, 3765, 9412, 16636],
     'excited': [38, 102, 1190, 1306, 3227, 5863],
     'solid': [37, 72, 2040, 6102, 22983, 31234],
     'dry': [78, 261, 3348, 3537, 4793, 12017],
     'maybe': [333, 861, 8287, 8889, 16807, 35177],
     'flavor': [34, 68, 892, 1097, 4027, 6118],
     'auction': [38, 170, 2803, 4432, 14129, 21572],
     'start': [200, 469, 4082, 5550, 17361, 27662],
     'session': [25, 75, 664, 1039, 3854, 5657],
     'works': [92, 230, 4419, 7681, 24561, 36983],
     'uberplay': [0, 2, 11, 32, 164, 209],
     'edition': [157, 361, 4015, 6621, 25016, 36170],
     'dune': [3, 7, 132, 143, 611, 896],
     'six': [69, 158, 1324, 1855, 4626, 8032],
     'suit': [14, 30, 568, 807, 1636, 3055],
     'faction': [20, 26, 488, 616, 2986, 4136],
     'event': [51, 176, 1403, 1547, 4616, 7793],
     'shuffled': [5, 21, 141, 199, 435, 801],
     'dealt': [32, 83, 598, 574, 1256, 2543],
     'five': [73, 214, 2143, 2598, 7589, 12617],
     'draft': [20, 63, 743, 1026, 3063, 4915],
     'four': [104, 351, 3700, 4959, 15051, 24165],
     'randomly': [118, 245, 1418, 1144, 2183, 5108],
     'assigned': [13, 15, 117, 146, 325, 616],
     'planets': [4, 29, 299, 425, 1033, 1790],
     'factions': [21, 40, 625, 863, 4598, 6147],
     'round': [181, 543, 5319, 7204, 19435, 32682],
     'planet': [17, 41, 346, 449, 1185, 2038],
     'may': [320, 678, 7029, 10628, 28734, 47389],
     'facedown': [0, 1, 26, 30, 65, 122],
     'stake': [3, 13, 57, 66, 197, 336],
     'claim': [22, 62, 522, 707, 1821, 3134],
     'towards': [50, 98, 1085, 1536, 3923, 6692],
     'favor': [29, 70, 578, 772, 2259, 3708],
     'protect': [12, 19, 147, 239, 623, 1040],
     'particular': [51, 98, 1217, 1462, 3636, 6464],
     'placing': [23, 64, 909, 1336, 3467, 5799],
     'revealed': [13, 48, 375, 549, 1302, 2287],
     'sorted': [1, 6, 58, 100, 339, 504],
     'executed': [11, 38, 332, 311, 979, 1671],
     'order': [178, 424, 4607, 6476, 17003, 28688],
     'thats': [362, 965, 7507, 7916, 20333, 37083],
     'gist': [1, 3, 48, 64, 130, 246],
     'random': [668, 2057, 14161, 11095, 17699, 45680],
     'opponents': [87, 278, 3371, 4946, 14346, 23028],
     'certain': [78, 233, 2699, 3682, 8953, 15645],
     'never': [1054, 2055, 11740, 10322, 29090, 54261],
     'know': [459, 1111, 8868, 9855, 27474, 47767],
     'plan': [73, 208, 1986, 2553, 8170, 12990],
     'implement': [15, 22, 128, 154, 453, 772],
     'scenario': [84, 191, 1533, 2158, 8545, 12511],
     'person': [200, 417, 3179, 3647, 8706, 16149],
     'smartest': [2, 2, 11, 24, 57, 96],
     'moves': [81, 227, 2539, 3503, 9811, 16161],
     'reality': [32, 50, 404, 371, 813, 1670],
     'variables': [3, 8, 102, 99, 508, 720],
     'swamp': [2, 2, 24, 36, 154, 218],
     'plans': [9, 48, 645, 867, 3161, 4730],
     'basis': [21, 25, 305, 312, 931, 1594],
     'go': [443, 1069, 9899, 12989, 36995, 61395],
     'got': [466, 1095, 9696, 11355, 32145, 54757],
     'boring': [869, 2404, 13163, 4934, 5571, 26941],
     'types': [42, 116, 1318, 1985, 5425, 8886],
     'strategy': [465, 1314, 12551, 17233, 60741, 92304],
     'seems': [308, 902, 13073, 17098, 33887, 65268],
     'panda': [8, 3, 53, 88, 393, 545],
     'save': [86, 177, 919, 815, 2120, 4117],
     'bamboo': [1, 7, 62, 55, 289, 414],
     'duration': [11, 30, 225, 316, 1244, 1826],
     'mind': [158, 359, 3195, 4216, 10178, 18106],
     'second': [201, 529, 4449, 5634, 17145, 27958],
     'assessment': [3, 6, 104, 139, 293, 545],
     'generic': [40, 70, 852, 722, 1201, 2885],
     'repetitive': [77, 306, 2973, 2732, 2933, 9021],
     'soulless': [17, 38, 282, 131, 141, 609],
     'euro': [92, 231, 3061, 4503, 19240, 27127],
     'limit': [25, 65, 766, 1151, 2842, 4849],
     'market': [82, 174, 2129, 2800, 8276, 13461],
     'timing': [12, 41, 697, 1127, 3986, 5863],
     'element': [48, 219, 2754, 3756, 10863, 17640],
     'mechanism': [71, 274, 3568, 4916, 13197, 22026],
     'falls': [40, 100, 1232, 1081, 1519, 3972],
     'flat': [50, 160, 2328, 1791, 2028, 6357],
     'rewarded': [9, 17, 185, 220, 820, 1251],
     'lame': [73, 189, 826, 367, 564, 2019],
     'intrigue': [6, 18, 264, 346, 1889, 2523],
     'sometimes': [81, 223, 2885, 4651, 14000, 21840],
     'fishing': [5, 10, 106, 119, 282, 522],
     'doge': [0, 1, 27, 33, 65, 126],
     'stack': [31, 84, 769, 935, 1953, 3772],
     'tile': [87, 285, 4553, 7281, 19004, 31210],
     'opponent': [90, 248, 2954, 3728, 11029, 18049],
     'valid': [14, 16, 149, 181, 505, 865],
     'man': [55, 145, 905, 1128, 4004, 6237],
     'ugly': [86, 175, 949, 762, 1448, 3420],
     'western': [4, 18, 219, 335, 1438, 2014],
     'trail': [8, 22, 131, 158, 541, 860],
     'explain': [55, 90, 861, 1678, 6749, 9433],
     'simple': [227, 793, 13249, 21974, 68786, 105029],
     'deep': [41, 92, 1444, 2781, 13786, 18144],
     'aside': [40, 70, 639, 785, 2193, 3727],
     'choice': [125, 308, 2642, 3036, 8116, 14227],
     'modifications': [6, 9, 83, 100, 316, 514],
     'read': [249, 471, 3238, 2968, 9277, 16203],
     'board': [823, 1557, 14634, 19057, 61361, 97432],
     'running': [43, 96, 895, 1136, 3426, 5596],
     'budding': [1, 1, 18, 22, 64, 106],
     'town': [14, 43, 396, 546, 1821, 2820],
     'excellent': [43, 111, 1799, 4008, 38885, 44846],
     'cardplay': [4, 9, 98, 139, 353, 603],
     'scale': [35, 106, 951, 1358, 4176, 6626],
     'noticeably': [0, 3, 29, 33, 98, 163],
     'rarely': [29, 109, 1081, 1272, 3262, 5753],
     'became': [41, 94, 794, 843, 2005, 3777],
     'grinding': [13, 32, 129, 83, 200, 457],
     'affair': [26, 28, 352, 383, 778, 1567],
     'thereafter': [2, 9, 36, 41, 110, 198],
     'simply': [296, 593, 4161, 3453, 9832, 18335],
     'takes': [244, 685, 6291, 7383, 21232, 35835],
     'conclusion': [26, 46, 360, 380, 765, 1577],
     'effort': [87, 178, 1497, 1391, 3178, 6331],
     'put': [254, 535, 4031, 4499, 12929, 22248],
     'peak': [0, 1, 32, 46, 221, 300],
     'interest': [93, 294, 3362, 2827, 5116, 11692],
     'middle': [45, 96, 1171, 1375, 3483, 6170],
     'anti': [47, 45, 433, 395, 888, 1808],
     'climactic': [1, 5, 117, 126, 238, 487],
     'theory': [13, 66, 560, 477, 697, 1813],
     'reasonable': [21, 45, 658, 986, 2413, 4123],
     'dam': [2, 2, 9, 14, 42, 69],
     'unfortunately': [108, 319, 3944, 3608, 5555, 13534],
     'tempted': [5, 22, 124, 182, 547, 880],
     'stab': [7, 7, 76, 114, 315, 519],
     'streamlining': [3, 10, 139, 147, 421, 720],
     'classic': [58, 274, 4021, 5708, 19596, 29657],
     'video': [59, 112, 768, 942, 3640, 5521],
     'fans': [71, 111, 1029, 1222, 3811, 6244],
     'thematic': [54, 175, 2136, 3328, 13877, 19570],
     'storm': [8, 14, 145, 170, 953, 1290],
     'moving': [72, 229, 2344, 2816, 7101, 12562],
     'burying': [3, 1, 10, 19, 41, 74],
     'ancient': [13, 36, 351, 520, 1725, 2645],
     'city': [44, 141, 1592, 2752, 9035, 13564],
     'deeper': [4, 27, 429, 922, 3434, 4816],
     'dirt': [7, 7, 92, 91, 292, 489],
     'successfully': [6, 22, 202, 258, 899, 1387],
     'escape': [51, 66, 548, 803, 2357, 3825],
     'buried': [15, 29, 188, 94, 263, 589],
     'desert': [13, 36, 331, 546, 1705, 2631],
     'young': [90, 199, 1843, 1538, 2968, 6638],
     'kid': [149, 458, 3889, 2435, 3203, 10134],
     'thomas': [3, 3, 21, 28, 70, 125],
     'train': [33, 86, 998, 1498, 5021, 7636],
     'eh': [14, 42, 766, 395, 344, 1561],
     'bits': [98, 224, 2554, 3130, 7394, 13400],
     'fact': [196, 499, 3802, 4110, 13069, 21676],
     'roll': [676, 1530, 8877, 7052, 12017, 30152],
     'move': [462, 1289, 8972, 8655, 18247, 37625],
     'fight': [68, 163, 1433, 1592, 5039, 8295],
     'pleasantly': [0, 5, 112, 337, 1427, 1881],
     'surprised': [18, 53, 683, 1302, 5269, 7325],
     'billed': [3, 3, 42, 38, 58, 144],
     'pick': [119, 280, 3246, 4790, 14661, 23096],
     'deliver': [25, 66, 894, 1248, 2899, 5132],
     'strikes': [4, 24, 226, 272, 772, 1298],
     'tiny': [41, 139, 1060, 1188, 3116, 5544],
     'management': [28, 120, 1718, 3040, 12091, 16997],
     'biggest': [67, 111, 1324, 1617, 3747, 6866],
     'involve': [12, 37, 309, 313, 806, 1477],
     'whether': [76, 216, 1833, 2340, 5686, 10151],
     'burn': [69, 51, 267, 358, 1292, 2037],
     'animals': [20, 33, 578, 998, 2118, 3747],
     'canoes': [0, 1, 18, 39, 73, 131],
     'available': [77, 202, 2386, 3164, 10881, 16710],
     'bide': [0, 0, 10, 12, 23, 45],
     'opportunity': [35, 135, 1080, 1317, 3925, 6492],
     'future': [53, 111, 1183, 1751, 6284, 9382],
     'sadly': [45, 130, 1187, 980, 1856, 4198],
     'completing': [6, 20, 237, 339, 1112, 1714],
     'small': [170, 412, 4706, 6614, 20022, 31924],
     'canoe': [0, 0, 26, 30, 67, 123],
     'loads': [14, 24, 268, 411, 2627, 3344],
     'compete': [13, 24, 339, 581, 1518, 2475],
     'loading': [3, 6, 43, 61, 154, 267],
     'high': [223, 509, 5555, 6822, 21801, 34910],
     'quicky': [0, 1, 9, 16, 37, 63],
     'presents': [3, 18, 190, 308, 1164, 1683],
     'choices': [205, 538, 4759, 5462, 18817, 29781],
     'abstract': [85, 365, 5883, 8336, 18032, 32701],
     'especially': [105, 336, 4612, 6852, 20426, 32331],
     'laying': [20, 72, 1319, 2419, 6436, 10266],
     'farming': [6, 17, 132, 185, 996, 1336],
     'depth': [51, 165, 2427, 4032, 17983, 24658],
     'scales': [4, 11, 200, 541, 3668, 4424],
     'deepest': [0, 1, 34, 75, 404, 514],
     'engaging': [19, 91, 1250, 1676, 6838, 9874],
     'gets': [284, 710, 7204, 8520, 21810, 38528],
     'throat': [4, 15, 158, 340, 1503, 2020],
     'unless': [155, 321, 2589, 2307, 4571, 9943],
     'actively': [27, 61, 334, 275, 590, 1287],
     'avoid': [199, 350, 1708, 1471, 3672, 7400],
     'blocking': [15, 55, 531, 878, 2272, 3751],
     'interplay': [1, 5, 61, 94, 448, 609],
     'jostling': [0, 1, 15, 31, 66, 113],
     'position': [47, 145, 1293, 1606, 4212, 7303],
     'expanded': [2, 6, 136, 183, 777, 1104],
     'tons': [20, 82, 637, 839, 5234, 6812],
     'recommended': [17, 66, 630, 1140, 8686, 10539],
     'baseline': [4, 7, 17, 21, 52, 101],
     'outdoor': [1, 2, 26, 22, 88, 139],
     'table': [221, 503, 5600, 8572, 27297, 42193],
     'usual': [24, 39, 722, 1031, 3001, 4817],
     'zombies': [46, 168, 1058, 1039, 2436, 4747],
     'problems': [94, 221, 2088, 2045, 4272, 8720],
     'otherwise': [113, 291, 3134, 3699, 8122, 15359],
     'fan': [124, 412, 5261, 5279, 12437, 23513],
     'knizia': [17, 78, 1115, 1644, 4868, 7722],
     'aid': [15, 27, 310, 465, 1495, 2312],
     'downloadable': [0, 4, 15, 25, 121, 165],
     'playable': [79, 151, 1347, 1663, 5590, 8830],
     'twice': [98, 261, 2422, 2581, 6779, 12141],
     'shaping': [0, 0, 18, 16, 73, 107],
     'result': [100, 196, 1849, 1842, 4225, 8212],
     'buying': [109, 177, 1538, 1842, 5668, 9334],
     'part': [219, 506, 5387, 6364, 16601, 29077],
     'pack': [26, 52, 620, 903, 4049, 5650],
     'london': [4, 5, 144, 220, 765, 1138],
     'competition': [17, 49, 654, 1077, 3523, 5320],
     'monster': [68, 140, 1212, 1518, 4263, 7201],
     'factory': [9, 32, 245, 271, 963, 1520],
     'favorite': [62, 171, 2424, 4557, 35039, 42253],
     'donald': [1, 7, 33, 34, 101, 176],
     'dominion': [48, 105, 1546, 2019, 6914, 10632],
     'monsters': [71, 171, 1490, 1831, 5340, 8903],
     'option': [40, 91, 944, 1300, 3779, 6154],
     'adults': [47, 139, 2064, 2693, 5752, 10695],
     'scrabble': [27, 92, 1034, 1185, 1963, 4301],
     'simplest': [3, 12, 94, 146, 494, 749],
     'must': [191, 417, 3482, 5440, 21442, 30972],
     'say': [365, 731, 5728, 6368, 19849, 33041],
     'wait': [129, 293, 1638, 1597, 9624, 13281],
     'word': [105, 238, 2589, 3048, 5970, 11950],
     'letters': [18, 57, 523, 626, 1090, 2314],
     'wants': [58, 160, 1295, 1246, 3050, 5809],
     'smart': [14, 45, 344, 654, 2704, 3761],
     'mathy': [4, 9, 256, 344, 808, 1421],
     'mrs': [0, 2, 28, 42, 188, 260],
     'english': [46, 63, 713, 1076, 3870, 5768],
     'students': [3, 5, 150, 230, 545, 933],
     'teaching': [44, 75, 620, 776, 2917, 4432],
     'tool': [24, 36, 280, 249, 681, 1270],
     'gamer': [76, 133, 1257, 1947, 6860, 10273],
     'geeks': [8, 13, 123, 155, 512, 811],
     'dexterity': [55, 191, 2380, 3027, 5609, 11262],
     'cheaper': [13, 32, 200, 281, 770, 1296],
     'smaller': [14, 32, 718, 1175, 3255, 5194],
     'shot': [44, 93, 1013, 1013, 2220, 4383],
     'cubes': [81, 152, 1786, 2367, 5787, 10173],
     'rinse': [17, 65, 287, 173, 158, 700],
     'repeat': [78, 193, 1204, 940, 1708, 4123],
     'sounds': [51, 115, 907, 847, 1914, 3834],
     'dull': [179, 600, 3838, 1681, 1929, 8227],
     'liken': [0, 4, 16, 15, 71, 106],
     'splendor': [5, 9, 292, 404, 1281, 1991],
     'micro': [20, 27, 268, 370, 898, 1583],
     'ends': [95, 249, 2005, 2318, 4426, 9093],
     'progress': [31, 90, 637, 664, 1799, 3221],
     'sense': [221, 408, 3189, 3179, 9558, 16555],
     'accomplishment': [2, 16, 124, 111, 498, 751],
     'rating': [708, 1122, 11610, 17416, 44292, 75148],
     'changed': [54, 66, 640, 715, 2134, 3609],
     'offensive': [49, 60, 278, 305, 836, 1528],
     'power': [93, 209, 2510, 3391, 11621, 17824],
     'weaker': [7, 24, 311, 330, 861, 1533],
     'unit': [48, 121, 804, 953, 3081, 5007],
     'damaged': [20, 15, 130, 231, 482, 878],
     'balls': [13, 24, 232, 235, 408, 912],
     'wall': [26, 70, 348, 525, 1268, 2237],
     'someone': [424, 878, 6432, 6586, 14534, 28854],
     'hold': [56, 140, 1782, 2099, 4775, 8852],
     'prescribed': [2, 0, 26, 29, 45, 102],
     'reinforcement': [0, 10, 59, 69, 324, 462],
     'weather': [3, 18, 179, 265, 927, 1392],
     'chance': [198, 493, 3991, 4133, 11217, 20032],
     'ending': [42, 96, 819, 992, 2322, 4271],
     'victory': [131, 324, 3674, 5250, 20398, 29777],
     'gm': [3, 16, 85, 118, 362, 584],
     'heavily': [31, 90, 960, 1056, 2174, 4311],
     'sheer': [21, 39, 286, 328, 1384, 2058],
     'scope': [4, 18, 274, 355, 1367, 2018],
     'breathtaking': [0, 3, 6, 15, 103, 127],
     'conceptually': [6, 8, 52, 46, 109, 221],
     'ultimate': [17, 27, 303, 386, 2091, 2824],
     'space': [172, 378, 3629, 4941, 15439, 24559],
     'opera': [1, 4, 48, 54, 257, 364],
     'something': [489, 1174, 11625, 12387, 24341, 50016],
     'unsure': [6, 7, 114, 295, 608, 1030],
     'ti': [1, 8, 63, 85, 385, 542],
     'reading': [142, 201, 1576, 1535, 4874, 8328],
     'fourth': [13, 23, 253, 276, 784, 1349],
     'decided': [112, 210, 1413, 1332, 3280, 6347],
     'finally': [118, 266, 1885, 2425, 10079, 14773],
     'plunge': [0, 3, 15, 19, 140, 177],
     'story': [126, 252, 2354, 2680, 10490, 15902],
     'concerns': [7, 3, 109, 250, 791, 1160],
     'glad': [50, 120, 865, 1133, 4669, 6837],
     'plastic': [123, 169, 1449, 1605, 4319, 7665],
     'stock': [47, 88, 1158, 1657, 5068, 8018],
     'thinner': [0, 2, 23, 28, 97, 150],
     'id': [317, 908, 10478, 12695, 19481, 43879],
     'fantasy': [63, 156, 1566, 1998, 7313, 11096],
     'flights': [1, 4, 44, 55, 145, 249],
     'epic': [41, 62, 627, 896, 5665, 7291],
     'creates': [14, 36, 453, 779, 3379, 4661],
     'memorable': [5, 8, 307, 368, 1009, 1697],
     'moments': [17, 48, 800, 1120, 3735, 5720],
     'borders': [4, 14, 116, 124, 354, 612],
     'established': [6, 12, 99, 117, 311, 545],
     'neighbors': [6, 12, 124, 200, 626, 968],
     'quickly': [140, 377, 5290, 7321, 19488, 32616],
     'combat': [205, 494, 4881, 5406, 17226, 28212],
     'goes': [99, 280, 2437, 3000, 7929, 13745],
     'keeps': [21, 77, 712, 1309, 7163, 9282],
     'ridiculous': [90, 145, 719, 516, 1253, 2723],
     'covering': [3, 8, 142, 195, 568, 916],
     'battles': [39, 115, 1295, 1752, 6553, 9754],
     'buildup': [1, 0, 54, 58, 158, 271],
     'yeah': [84, 154, 924, 766, 2095, 4023],
     'veterans': [1, 5, 41, 90, 491, 628],
     'components': [330, 862, 7985, 10971, 35554, 55702],
     'track': [87, 307, 3157, 4219, 11947, 19717],
     'regardless': [20, 50, 458, 541, 1883, 2952],
     'admire': [3, 11, 162, 245, 497, 918],
     'rock': [37, 104, 601, 571, 1267, 2580],
     'exactly': [85, 189, 1791, 1926, 5145, 9136],
     'nimble': [0, 0, 9, 9, 65, 83],
     'awhile': [10, 17, 364, 625, 1269, 2285],
     'point': [397, 914, 7901, 9026, 22758, 40996],
     'whose': [25, 46, 367, 398, 974, 1810],
     'tastes': [18, 79, 1501, 1646, 1583, 4827],
     'likes': [48, 103, 1397, 1863, 5134, 8545],
     'sort': [120, 341, 3658, 3973, 7264, 15356],
     'finds': [9, 15, 139, 181, 491, 835],
     'shine': [4, 12, 365, 730, 2234, 3345],
     'terms': [28, 97, 1169, 1552, 4567, 7413],
     'weekend': [10, 21, 129, 198, 1251, 1609],
     'boy': [35, 66, 394, 377, 1420, 2292],
     'proposition': [3, 3, 32, 24, 74, 136],
     'said': [191, 323, 3100, 3856, 10211, 17681],
     'lore': [6, 13, 113, 139, 909, 1180],
     'pushing': [21, 49, 512, 642, 1588, 2812],
     'around': [307, 843, 7512, 9060, 23885, 41607],
     'gobbling': [1, 1, 9, 12, 20, 43],
     'systems': [15, 41, 499, 563, 2145, 3263],
     'fleets': [3, 9, 111, 117, 419, 659],
     'strike': [16, 52, 360, 382, 1124, 1934],
     'moment': [40, 79, 714, 978, 3978, 5789],
     'came': [145, 250, 2248, 2508, 6521, 11672],
     'expectations': [16, 33, 552, 651, 1446, 2698],
     'change': [124, 253, 2630, 4065, 11172, 18244],
     'tis': [1, 1, 22, 36, 104, 164],
     'younger': [11, 78, 1508, 1721, 2600, 5918],
     'sibling': [0, 3, 59, 59, 243, 364],
     'runewars': [1, 3, 18, 70, 178, 270],
     'ways': [81, 203, 2263, 3412, 14767, 20726],
     'lend': [3, 8, 80, 105, 355, 551],
     'lower': [71, 117, 1239, 1571, 3593, 6591],
     'counts': [13, 53, 587, 995, 4157, 5805],
     'itching': [0, 0, 38, 95, 217, 350],
     'ol': [3, 10, 72, 104, 277, 466],
     'scratch': [9, 17, 280, 388, 950, 1644],
     'itch': [3, 15, 338, 661, 2224, 3241],
     'graphics': [38, 112, 1143, 1431, 3632, 6356],
     'fine': [79, 256, 3798, 5256, 8891, 18280],
     'barely': [101, 252, 974, 675, 1454, 3456],
     'upside': [6, 23, 182, 209, 432, 852],
     'drinking': [34, 86, 642, 587, 1040, 2389],
     'beers': [0, 9, 141, 169, 346, 665],
     'basic': [90, 294, 3735, 4230, 9418, 17767],
     'captured': [4, 10, 151, 223, 751, 1139],
     'feeling': [50, 225, 2953, 3900, 11048, 18176],
     'gunfight': [0, 0, 12, 18, 45, 75],
     'somethings': [3, 2, 51, 45, 70, 171],
     'switching': [5, 8, 123, 140, 376, 652],
     'info': [17, 29, 258, 381, 993, 1678],
     'advanced': [31, 99, 1228, 1956, 6280, 9594],
     'added': [57, 170, 1717, 2464, 8113, 12521],
     'complexity': [46, 179, 1652, 2258, 9139, 13274],
     'wills': [2, 0, 3, 8, 33, 46],
     'low': [186, 358, 3592, 3803, 9103, 17042],
     'moved': [21, 71, 746, 877, 1932, 3647],
     'captures': [4, 8, 232, 477, 2992, 3713],
     'show': [93, 182, 1462, 1709, 4886, 8332],
     'decent': [83, 240, 5324, 10276, 9504, 25427],
     'lasted': [14, 72, 389, 371, 684, 1530],
     'rares': [6, 2, 40, 49, 176, 273],
     'says': [92, 152, 887, 915, 2562, 4608],
     'anything': [386, 811, 5890, 4823, 9002, 20912],
     'information': [78, 220, 1990, 2130, 5332, 9750],
     'mostly': [97, 291, 3750, 3815, 8102, 16055],
     'multiplayer': [28, 110, 1313, 1466, 4528, 7445],
     'experienced': [34, 96, 1111, 1591, 6836, 9668],
     'remember': [128, 423, 3433, 3341, 6608, 13933],
     'expansions': [101, 179, 2397, 4814, 27311, 34802],
     'mixed': [11, 67, 706, 1186, 3154, 5124],
     'pre': [59, 102, 965, 932, 2991, 5049],
     'school': [64, 110, 1228, 1334, 2899, 5635],
     'entirely': [94, 208, 1368, 1249, 2487, 5406],
     'started': [76, 216, 1610, 2015, 6779, 10696],
     'found': [227, 571, 6423, 6529, 14483, 28233],
     'difficult': [94, 366, 4317, 6208, 18876, 29861],
     'across': [54, 99, 994, 1202, 3633, 5982],
     'later': [85, 203, 2224, 2775, 7996, 13283],
     'called': [109, 167, 975, 941, 2373, 4565],
     'quits': [3, 10, 41, 23, 31, 108],
     'yahtzee': [39, 128, 1428, 1392, 1830, 4817],
     'manipulating': [2, 7, 140, 179, 549, 877],
     'suffer': [29, 52, 493, 642, 1825, 3041],
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

### 8.Referrences

Professor Mr. Park's data mining Naïve Bayes lecture.

https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

https://en.wikipedia.org/wiki/Naive_Bayes_classifier

https://gist.github.com/sebleier/554280

https://blog.csdn.net/longxinchen_ml/article/details/50597149
