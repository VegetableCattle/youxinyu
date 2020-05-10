---
title: Data Mining Assignment 03 - NBC
subtitle: The content of this blog is to implement Naive Bayes Classifier algorithm by using Jupyter notebook and [Large Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/).[Git](https://github.com/VegetableCattle/youxinyu/tree/master/static/files/li_03.ipynb).[Download file](https://yongli.netlify.com/files/li_03.ipynb)
summary: The content of this blog is to implement Naive Bayes Classifier algorithm by using Jupyter notebook and [Large Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/).
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
```python
import os
import numpy as np
from sys import path
import re
import random
```

### read file


```python
def readFile(dir_path, label, data):
    for file_name in os.listdir(dir_path):
        f=open(dir_path+file_name,'r',encoding='utf8')
        lines=f.readlines()
        data.append([lines,label])
    random.shuffle(data)
```

## a.Divide the dataset as train, development and test. 


```python
# Split dataset to k folds
def crossValidationSplit(data, k_folds):
    data_split = list()
    data_copy = list(data)
    size = int(len(data) / k_folds)
    for _ in range(k_folds):
        fold = list()
        while len(fold) < size:
            k = random.randrange(len(data_copy))
            fold.append(data_copy.pop(k))
        data_split.append(fold)
    return data_split

def splitDataToTrainAndDev(dataset, k_folds):
    folds = crossValidationSplit(dataset, k_folds)
    train_set, dev_set = [], []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        dev_set = list()
        for row in fold:
                row_copy = list(row)
                dev_set.append(row_copy)
        break
    return train_set, dev_set

dataset_neg, dataset_pos = [], []
readFile(path[0] + '/aclImdb/train/neg/', int(-1), dataset_neg)
readFile(path[0] + '/aclImdb/train/pos/', int(1), dataset_pos)

n_folds = 5
train_neg, dev_neg = splitDataToTrainAndDev(dataset_neg, n_folds)
train_pos, dev_pos = splitDataToTrainAndDev(dataset_pos, n_folds)


test_neg, test_pos = [], []
readFile(path[0] + '/aclImdb/test/neg/', int(-1), test_neg)
readFile(path[0] + '/aclImdb/test/pos/', int(1), test_pos)


train = train_neg + train_pos
dev = dev_neg + dev_pos
test = test_neg + test_pos

print('Train dataset size: ' + str(len(train)))
print('Dev dataset size: ' + str(len(dev)))
print('Test dataset size: ' + str(len(test)))
```

    Train dataset size: 20000
    Dev dataset size: 5000
    Test dataset size: 25000


## b.Build a vocabulary as list. 

\[‘the’ ‘I’ ‘happy’ … \] 

You may omit rare words for example if the occurrence is less than five times 

A reverse index as the key value might be handy 

{“the”: 0, “I”:1, “happy”:2 , … }



```python
def SegmentLineToWords(string):
    string=string.replace('<br />', '')
    return set([x.lower() for x in re.split(r'[\s|,|;|.|/|\[|\]|;|\!|?|\'|\\|\)|\(|\"|@|&|#|-|*|%|>|<|^|-]\s*',string.strip()) if x])

def buildVocabularyList(dataset):
    dict_list = {} #{'word':[neg_count, pos_count]}
    for row in dataset:
        words = set() #Words that appear multiple times in the same comment are counted only once
        words = words.union(SegmentLineToWords(str(row[0])))
        for word in words:
            if word not in dict_list:
                dict_list[word] = [0,0]
            if row[1] == -1:
                dict_list[word][0] += 1
            else:
                dict_list[word][1] += 1
    for word in list(dict_list.keys()):
        if dict_list[word][0] + dict_list[word][1]<5:
            del dict_list[word]
    return dict_list
train_dict = buildVocabularyList(train)
train_dict
```




    {'apparent': [141, 102],
     'matthew': [33, 43],
     'every': [1328, 1281],
     'camera': [691, 494],
     'can': [4008, 3686],
     ...}



## c.Calculate the following probability

Probability of the occurrence

P\[“the”\] = num of documents containing ‘the’ / num of all documents


```python
def getProbabilityOfOccurrence(word):
    if word not in train_dict:
        return 0
    else: 
        return (train_dict[word][0] + train_dict[word][1])/(len(train))
print("P[“the”] = " + str(getProbabilityOfOccurrence("the")))
```

    P[“the”] = 0.992


Conditional probability based on the sentiment

P\[“the” | Positive\]  = # of positive documents containing “the” / num of all positive review documents



```python
def getPosConditionalProbability(word):
    if word not in train_dict:
        return 0
    else:
        return train_dict[word][1]/(len(train_pos))
def getNegConditionalProbability(word):
    if word not in train_dict:
        return 0
    else:
        return train_dict[word][0]/(len(train_neg))

print("P[“the” | Positive] = " + str(getPosConditionalProbability("the")))
print("P[“the” | negative] = " + str(getNegConditionalProbability("the")))
```

    P[“the” | Positive] = 0.9908
    P[“the” | negative] = 0.9932


## d.Calculate accuracy using dev dataset 


```python
def predict(review, smoothing_flag):
    words = set()
    words = words.union(SegmentLineToWords(review))
    pos_probability = 1
    neg_probability = 1
    for word in words:
        if smoothing_flag == 1:
            pos_probability *= getPosConditionalProbabilityUsingSmoothing(word)
            neg_probability *= getNegConditionalProbabilityUsingSmoothing(word)
            
        else:
            #print(word)
            #print("getPosConditionalProbability: " + str(getPosConditionalProbability(word)))
            #print("getNegConditionalProbability: " + str(getNegConditionalProbability(word)))
            pos_probability *= getPosConditionalProbability(word)
            neg_probability *= getNegConditionalProbability(word)
    #print("pos_probability: " + str(pos_probability))
    #print("neg_probability: " + str(neg_probability))
    
    return 1 if pos_probability >= neg_probability else -1

def accuracy_metric(test_dataset, smoothing_flag):
    correct = 0
    for row in test_dataset:
        #print( predict(str(row[0]), smoothing_flag))
        #print(row[1])
        if row[1] == predict(str(row[0]), smoothing_flag):
            correct += 1
    return correct / float(len(test_dataset)) * 100.0
```


```python
#train_dict
print('Accuracy: %.3f%%' % accuracy_metric(dev, 0))
```

    Accuracy: 54.740%


### Conduct five fold cross validation


```python
def evaluate_algorithm(pos_dataset, neg_dataset, k_folds, smoothing_flag):
    pos_folds = crossValidationSplit(pos_dataset, k_folds)
    neg_folds = crossValidationSplit(neg_dataset, k_folds)
    scores = list()
    for i in range(0,len(pos_folds)):
        train_pos = list(pos_folds)
        train_neg = list(neg_folds)
        
        train_pos.remove(pos_folds[i])
        train_neg.remove(neg_folds[i])
        
        train_pos = sum(train_pos, [])
        train_neg = sum(train_neg, [])
        
        dev_pos = list()
        for row in pos_folds[i]:
            row_copy = list(row)
            dev_pos.append(row_copy)
        dev_neg = list()
        for row in neg_folds[i]:
            row_copy = list(row)
            dev_neg.append(row_copy)
        
        train = train_pos + train_neg
        train_dict = buildVocabularyList(train)
        dev = dev_pos + dev_neg
        accuracy = accuracy_metric(dev, smoothing_flag)
        scores.append(accuracy)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
smoothing_flag = 0
evaluate_algorithm(dataset_pos, dataset_neg, 5, smoothing_flag)
```

    Scores: [56.16, 56.49999999999999, 56.720000000000006, 56.3, 56.02]
    Mean Accuracy: 56.340%


## e.Do following experiments

### Compare the effect of Smoothing


```python
lambda_value = 1
def getPosConditionalProbabilityUsingSmoothing(word):
    if word not in train_dict:
        return lambda_value/(2*lambda_value+len(train_pos))
    else:
        return (lambda_value + train_dict[word][1])/(2*lambda_value+len(train_pos))
def getNegConditionalProbabilityUsingSmoothing(word):
    if word not in train_dict:
        return lambda_value/(2*lambda_value+len(train_neg))
    else:
        return (lambda_value+train_dict[word][0])/(2*lambda_value+len(train_neg))
```


```python
smoothing_flag = 1

print('Accuracy not using Smoothing: %.3f%%' % accuracy_metric(dev, 0))
print('Accuracy by using Smoothing: %.3f%%' % accuracy_metric(dev, smoothing_flag))
```

    Accuracy not using Smoothing: 54.740%
    Accuracy by using Smoothing: 80.820%


### Derive Top 10 words that predicts positive and negative class
### P\[Positive| word\] 
P\[Positive| word\] = P\[word| Positive\] * P\[Positive\] / P\[word\]

P\[word\] = P\[word& Positive\] + P\[word& negative\]

Derive Top 10 words that predicts positive and negative class by using train data


```python
def getTop10UsingTrain(label):
    positive_list = []
    for word in list(train_dict.keys()):
        #value = ((train_dict[word][1] / len(train_pos)) * (len(train_pos) / (len(train)))) / (train_dict[word][1]/len(train_pos)*(len(train_pos) / len(train)) + train_dict[word][0])/len(train_neg)*(len(train_neg) / len(train))
        value = (train_dict[word][label] + lambda_value) / (train_dict[word][1] + lambda_value + train_dict[word][0] + lambda_value)
        positive_list.append([word,value])
    return positive_list
positive_list = np.array(getTop10UsingTrain(1))
positive_list = positive_list[np.lexsort(positive_list.T)]

negative_list = np.array(getTop10UsingTrain(0))
negative_list = negative_list[np.lexsort(negative_list.T)]
def printTop10(data_list):
    for i in range(1, len(data_list)):
        if i <= 10:
            print(data_list[-i][0])
        else:
            break
print("Top 10 words that predicts positive by using train data :")
printTop10(positive_list)
print("")
print("Top 10 words that predicts negative by using train data :")
printTop10(negative_list)
```

    Top 10 words that predicts positive by using train data :
    edie
    antwone
    din
    unpretentious
    philo
    mcintire
    gunga
    sabu
    miklos
    knockout
    
    Top 10 words that predicts negative by using train data :
    steaming
    boll
    uwe
    ajay
    hobgoblins
    kareena
    stinker
    slater
    savini
    ketchup


### As we can see,if only the training set and formula are used to count the data obtained, the performance is not good. So the in the follows i will use the dev and algorithm to get top 10 word


```python
def getDevPredictsList():
    predicts_list = {}
    correct = 0
    for row in dev:
        if row[1] == predict(str(row[0]), 1):
            words = set()
            words = words.union(SegmentLineToWords(str(row[0])))
            for word in words:
                if word not in predicts_list:
                    predicts_list[word] = [0,0]
                if row[1] == -1:
                    predicts_list[word][0] += 1
                else:
                    predicts_list[word][1] += 1
    return predicts_list
predicts_list = getDevPredictsList()
```


```python
def getTop10UsingDev(label):
    positive_list = []
    for word in list(predicts_list.keys()):
        #value = ((train_dict[word][1] / len(train_pos)) * (len(train_pos) / (len(train)))) / (train_dict[word][1]/len(train_pos)*(len(train_pos) / len(train)) + train_dict[word][0])/len(train_neg)*(len(train_neg) / len(train))
        #value = float(train_dict[word][1]) / float(len(train) * (train_dict[word][1] + train_dict[word][0]))
        value = (predicts_list[word][label] + lambda_value) / (predicts_list[word][1] + lambda_value + predicts_list[word][0] + lambda_value)
        positive_list.append([word,value])
    return positive_list
positive_list = np.array(getTop10UsingDev(1))
positive_list = positive_list[np.lexsort(positive_list.T)]

negative_list = np.array(getTop10UsingDev(0))
negative_list = negative_list[np.lexsort(negative_list.T)]

print("Top 10 words that predicts positive by using dev data :")
printTop10(positive_list)
print("")
print("Top 10 words that predicts negative by using dev data :")
printTop10(negative_list)
```

    Top 10 words that predicts positive by using dev data :
    freedom
    extraordinary
    remarkable
    italy
    powell
    poverty
    cleverly
    journey
    solo
    glorious
    
    Top 10 words that predicts negative by using dev data :
    lousy
    mst3k
    pointless
    dreck
    unwatchable
    hackneyed
    redeeming
    waste
    tiresome
    drivel


## f.Using the test dataset


```python
print('The accuracy by using Smoothing of test dataset: %.3f%%' % accuracy_metric(test, smoothing_flag))
```

    The accuracy by using Smoothing of test dataset: 79.240%

