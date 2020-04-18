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
     'chimp': [14, 2],
     'watching': [1732, 1332],
     'there': [4713, 3926],
     'line': [680, 558],
     'luckily': [49, 54],
     'to': [9497, 9306],
     'amazing': [194, 711],
     'project': [220, 130],
     'they': [4729, 3947],
     'man': [1413, 1817],
     'jackie': [25, 67],
     'ladies': [92, 119],
     'at': [5394, 4994],
     'is': [8852, 9083],
     'bond': [53, 134],
     'shortly': [46, 47],
     'classics': [71, 106],
     'develops': [37, 75],
     'finding': [104, 163],
     'call': [433, 269],
     'telling': [205, 224],
     'seemed': [576, 348],
     'tricks': [35, 63],
     'li': [27, 30],
     'and': [9614, 9722],
     'but': [7384, 7003],
     'hogan': [9, 6],
     'the': [9932, 9908],
     'a': [9686, 9654],
     'kid': [402, 323],
     'phone': [132, 87],
     'martial': [93, 87],
     'real': [1409, 1558],
     'for': [7187, 7147],
     'one': [5604, 5687],
     'when': [3530, 3702],
     'even': [3945, 2781],
     'obvious': [502, 295],
     't': [6775, 5336],
     'funky': [12, 18],
     'hulk': [17, 11],
     'picks': [49, 78],
     'got': [1338, 1094],
     'prove': [113, 95],
     'awesome': [79, 244],
     'learns': [47, 121],
     'still': [1517, 2034],
     'action': [868, 975],
     'artist': [89, 142],
     'monkeys': [25, 20],
     'seem': [831, 721],
     'posters': [29, 31],
     'acting': [2630, 1625],
     'such': [1623, 1602],
     'some': [4058, 3682],
     'like': [5012, 4274],
     'long': [1205, 1149],
     'becomes': [453, 519],
     'replacing': [19, 8],
     'odd': [208, 221],
     'leans': [3, 8],
     'i': [8257, 7635],
     'should': [1831, 1425],
     'perhaps': [556, 608],
     'using': [319, 277],
     'handle': [65, 66],
     'story': [2750, 3320],
     'affleck': [26, 10],
     'up': [3623, 3310],
     'from': [4636, 4770],
     'bit': [875, 1146],
     'untrained': [3, 2],
     'movies': [2239, 1976],
     'guys': [474, 344],
     'seeing': [728, 791],
     'because': [2710, 2253],
     'had': [3270, 2708],
     'help': [616, 683],
     'jet': [33, 35],
     'heck': [105, 67],
     'close': [454, 462],
     'ben': [135, 146],
     'good': [3831, 3853],
     'modine': [4, 5],
     'it': [8950, 8891],
     'stunt': [53, 38],
     'on': [6375, 6173],
     'sidekick': [42, 42],
     'two': [1902, 2105],
     'this': [9238, 8874],
     'girl': [851, 834],
     'strong': [235, 546],
     'come': [1134, 1109],
     'look': [1452, 1289],
     'have': [5964, 5315],
     'by': [4591, 4777],
     'way': [2397, 2449],
     'while': [1523, 1774],
     'he': [4000, 4266],
     'which': [3033, 3057],
     'doesn': [1610, 1301],
     'that': [8292, 7932],
     'wasn': [974, 568],
     'these': [1670, 1606],
     'possibility': [32, 49],
     'confidence': [25, 53],
     'was': [6845, 6047],
     'replaced': [63, 65],
     'figured': [100, 48],
     'an': [4781, 5000],
     'loose': [109, 111],
     'who': [4352, 4744],
     'second': [669, 682],
     'x': [94, 109],
     'be': [5975, 5350],
     'would': [3625, 2913],
     'helped': [108, 136],
     'm': [1795, 1225],
     'shakespearean': [13, 15],
     'school': [559, 457],
     'chan': [20, 51],
     'in': [8761, 8882],
     'movie': [6606, 5609],
     'maybe': [940, 645],
     'get': [2873, 2553],
     'of': [9492, 9487],
     'trained': [25, 51],
     'days': [430, 508],
     'or': [4581, 3679],
     'mention': [361, 266],
     'found': [891, 911],
     'required': [77, 61],
     'argue': [50, 39],
     'them': [2347, 2146],
     'figuring': [12, 11],
     'star': [629, 665],
     'monkey': [56, 24],
     'time': [3481, 3484],
     'thought': [1233, 1080],
     'jealous': [33, 56],
     'with': [6895, 7084],
     'dorky': [10, 7],
     'watched': [811, 778],
     'local': [299, 316],
     'copy': [197, 214],
     'after': [2368, 2282],
     'as': [6182, 6734],
     'were': [3014, 2420],
     'did': [2109, 1727],
     'seven': [91, 125],
     'might': [1143, 818],
     'ride': [106, 200],
     'sequences': [248, 253],
     'work': [1345, 1496],
     'pick': [180, 159],
     'are': [5461, 5648],
     'fiddle': [7, 3],
     'into': [2645, 2530],
     'bad': [3523, 1199],
     'obviously': [527, 321],
     'ground': [145, 127],
     'middle': [381, 341],
     'p': [113, 110],
     'deaf': [25, 23],
     'make': [2843, 2089],
     'born': [109, 163],
     'instance': [122, 94],
     'perception': [16, 29],
     'blatant': [64, 22],
     'student': [151, 111],
     'true': [574, 1003],
     'watch': [2226, 2134],
     'debate': [29, 22],
     'involved': [418, 381],
     'under': [492, 508],
     'towards': [221, 264],
     'all': [5286, 5193],
     'd': [1093, 806],
     'documentary': [192, 300],
     'o': [240, 284],
     'attempt': [530, 225],
     'w': [81, 109],
     'slightly': [194, 239],
     'common': [193, 193],
     'capital': [29, 33],
     'leaning': [12, 4],
     'not': [6315, 5614],
     'went': [604, 465],
     'biased': [27, 24],
     'didn': [1680, 1066],
     'take': [1232, 1155],
     'topic': [59, 51],
     'opinion': [309, 388],
     'account': [66, 71],
     's': [7222, 7265],
     'bias': [16, 23],
     'l': [118, 117],
     'wrong': [780, 505],
     'situations': [182, 194],
     'v': [79, 72],
     'out': [4414, 4104],
     'their': [2692, 2913],
     'what': [4183, 3754],
     'current': [86, 114],
     'virtually': [94, 62],
     'other': [2602, 2814],
     'class': [287, 309],
     'right': [1079, 1126],
     'also': [2113, 3059],
     'e': [223, 221],
     'am': [986, 854],
     'forced': [290, 220],
     'unbiased': [7, 4],
     'no': [3972, 2578],
     'hearing': [89, 90],
     'later': [613, 836],
     'people': [2592, 2406],
     'film': [5476, 5670],
     'directed': [407, 474],
     'been': [2909, 2390],
     'start': [696, 538],
     'jeremy': [36, 58],
     'could': [2773, 1988],
     'think': [2209, 2114],
     'wouldn': [506, 269],
     'waste': [970, 73],
     'actor': [789, 798],
     'things': [1166, 1215],
     'don': [3119, 2141],
     'about': [4332, 4036],
     'if': [4701, 3843],
     'has': [3750, 4277],
     'where': [1968, 1868],
     'along': [572, 738],
     'you': [5653, 5280],
     'usually': [359, 337],
     'part': [1234, 1314],
     'those': [1445, 1617],
     'plot': [2521, 1500],
     'said': [844, 670],
     'entertaining': [449, 629],
     'read': [704, 571],
     'unfortunately': [682, 306],
     'london': [110, 157],
     'laughs': [263, 210],
     'performance': [626, 1182],
     'my': [3195, 3271],
     'disappointed': [448, 268],
     'potential': [288, 155],
     'solid': [104, 271],
     'comment': [255, 204],
     'horrible': [687, 119],
     'lead': [491, 456],
     'great': [1684, 3397],
     'need': [674, 601],
     'well': [2578, 3473],
     'least': [1386, 818],
     'so': [5042, 4311],
     'female': [391, 281],
     'really': [3225, 2887],
     'penultimate': [6, 8],
     'pacino': [39, 66],
     'midnight': [37, 64],
     'difference': [145, 149],
     'numbers': [142, 127],
     'minutes': [1337, 557],
     'fails': [361, 92],
     'anything': [1288, 776],
     'garden': [30, 35],
     'only': [3756, 2909],
     'convoluted': [60, 36],
     'standout': [8, 38],
     'tale': [171, 358],
     'same': [1379, 1316],
     'interesting': [1100, 1017],
     'than': [2978, 2683],
     'usual': [300, 407],
     'completely': [813, 524],
     '40': [131, 125],
     'self': [442, 385],
     'cast': [1122, 1468],
     'hardly': [274, 162],
     'al': [99, 112],
     'off': [2098, 1551],
     'southern': [58, 58],
     'cusack': [23, 29],
     'paycheck': [29, 1],
     'credibility': [67, 35],
     'pile': [128, 27],
     'through': [1710, 1521],
     'better': [2145, 1520],
     'added': [160, 174],
     'just': [4748, 3681],
     'end': [1885, 1723],
     'little': [1890, 1992],
     'each': [635, 1044],
     'thing': [1869, 1123],
     'throw': [198, 103],
     'mindless': [84, 35],
     'romantic': [183, 363],
     'act': [486, 360],
     'his': [3961, 4584],
     'then': [2592, 1963],
     'deliver': [136, 113],
     'done': [1020, 1058],
     'won': [543, 660],
     'nauseous': [13, 0],
     'revealing': [40, 55],
     'times': [991, 1257],
     'john': [482, 761],
     'does': [1741, 1873],
     'cares': [107, 81],
     'loses': [82, 97],
     'scene': [1553, 1486],
     'silliness': [38, 23],
     'profound': [38, 73],
     'flatter': [7, 1],
     'final': [420, 493],
     'brings': [156, 323],
     'first': [2438, 2666],
     'inaccurate': [30, 9],
     'makes': [1246, 1587],
     'walking': [197, 114],
     'preaching': [16, 17],
     'character': [1920, 1944],
     'mayor': [17, 24],
     'preachy': [28, 23],
     'supporting': [216, 432],
     'without': [1111, 1179],
     'hour': [548, 278],
     'search': [92, 138],
     'shtick': [16, 6],
     'crap': [595, 93],
     'last': [927, 1039],
     'speaking': [167, 137],
     'evil': [452, 387],
     'seems': [1299, 1034],
     'collecting': [12, 18],
     'accent': [216, 125],
     'law': [134, 196],
     'junk': [123, 25],
     'interest': [420, 352],
     'boy': [437, 523],
     'noticeably': [9, 11],
     'preach': [10, 11],
     'more': [3528, 3719],
     'supposed': [812, 230],
     'far': [1068, 941],
     'several': [444, 548],
     'righteous': [12, 16],
     'another': [1523, 1370],
     'noise': [65, 29],
     'her': [2408, 2730],
     'vehicle': [102, 75],
     'committed': [68, 81],
     'talking': [373, 298],
     'tracking': [20, 34],
     'foot': [92, 68],
     'spend': [229, 152],
     'thugs': [45, 30],
     'woman': [827, 879],
     'motion': [170, 164],
     '4': [623, 288],
     'opportunity': [138, 158],
     'kill': [488, 344],
     'amongst': [53, 58],
     'heels': [26, 24],
     '10': [1314, 1338],
     'minute': [345, 234],
     'home': [574, 650],
     'run': [484, 404],
     'witnessed': [34, 30],
     'she': [2104, 2259],
     'oh': [741, 271],
     'hear': [275, 290],
     'yea': [16, 3],
     'arguing': [28, 20],
     'themselves': [423, 430],
     'leave': [403, 411],
     'relationship': [213, 437],
     'bumper': [3, 6],
     'head': [596, 463],
     '5': [498, 422],
     'crime': [200, 298],
     'yet': [810, 1000],
     'nice': [614, 718],
     'until': [606, 671],
     'person': [577, 510],
     'message': [236, 313],
     'men': [475, 650],
     'director:': [8, 3],
     'someone': [997, 587],
     'abusive': [29, 39],
     'find': [1273, 1463],
     'order': [306, 363],
     'away': [1042, 905],
     'sit': [347, 196],
     'brutally': [35, 38],
     'herself': [211, 333],
     'bat': [52, 25],
     'bizarre': [201, 155],
     'chinese': [65, 104],
     'confusing': [187, 79],
     'flow': [54, 76],
     'beware': [21, 20],
     'video': [673, 434],
     'brief': [146, 169],
     'love': [1308, 2263],
     'ok': [482, 225],
     'looks': [1000, 596],
     'visually': [75, 123],
     'smoothly': [9, 20],
     'badly': [382, 77],
     'shocked': [81, 79],
     'understand': [607, 550],
     'epic': [74, 140],
     've': [1684, 1470],
     'follow': [256, 325],
     'recommend': [491, 771],
     'stunning': [67, 229],
     'translation': [46, 35],
     're': [1537, 1164],
     'looking': [980, 736],
     'prologue': [15, 13],
     'films': [1684, 1969],
     '2001': [48, 65],
     'development': [278, 190],
     'introduction': [57, 68],
     'characters': [2057, 2037],
     'place': [808, 873],
     '80s': [107, 90],
     'too': [2376, 2177],
     'appreciate': [134, 259],
     'promise': [89, 66],
     'released': [294, 430],
     'over': [1957, 1821],
     'will': [2234, 2811],
     'much': [3008, 2721],
     'scenes': [1649, 1453],
     'hard': [965, 932],
     'fantasy': [143, 234],
     'filmed': [272, 278],
     'secondly': [60, 25],
     'carlitos': [5, 0],
     'intelligent': [158, 254],
     'quote:': [1, 5],
     'modern': [246, 386],
     'mean': [767, 395],
     'full': [582, 683],
     'guess': [664, 310],
     'rise': [75, 104],
     'huge': [368, 309],
     'carlito': [17, 5],
     'duration': [35, 21],
     'me': [3104, 2771],
     'most': [2410, 2787],
     'acted': [255, 255],
     'named': [239, 287],
     'hooked': [33, 72],
     'whit': [7, 5],
     'prequel': [38, 15],
     'seen': [2120, 2158],
     'say': [1929, 1522],
     '50': [195, 133],
     'power': [257, 352],
     'anyone': [1000, 868],
     'do': [2967, 2356],
     'change': [284, 421],
     'its': [1858, 2263],
     'yes': [575, 508],
     'blockbuster': [83, 62],
     'name': [643, 467],
     'budget': [770, 411],
     'title': [575, 470],
     'know': [2041, 1780],
     'peace': [52, 93],
     'way:': [18, 7],
     'something': [1864, 1378],
     'meaning': [185, 183],
     'yelling': [50, 21],
     'hell': [475, 262],
     'very': [3042, 3875],
     'q': [36, 38],
     'day': [778, 1013],
     'brigante': [7, 0],
     'lower': [99, 64],
     'classic': [453, 792],
     'wont': [41, 40],
     'asian': [101, 53],
     'fun': [661, 1063],
     'minimal': [63, 34],
     'group': [394, 306],
     'jungle': [53, 53],
     'short': [596, 637],
     'gore': [446, 186],
     'party': [168, 192],
     '2': [1159, 583],
     'hunter': [80, 65],
     'trash': [269, 92],
     'city': [259, 454],
     'director': [1502, 1318],
     'labeled': [17, 16],
     'sleazy': [54, 61],
     'italian': [135, 170],
     'race': [109, 118],
     'romp': [26, 41],
     'worse': [838, 166],
     'mexico': [45, 67],
     'black': [596, 598],
     'tribe': [27, 23],
     'kids': [519, 498],
     'cannibal': [45, 5],
     'somewhere': [232, 143],
     'horror': [1060, 616],
     'kidnapped': [53, 54],
     'suppose': [191, 104],
     'lot': [1241, 1360],
     'halloween': [72, 54],
     'cut': [429, 263],
     'superior': [92, 137],
     'ultra': [67, 42],
     'garb': [5, 4],
     'franco': [27, 15],
     'scares': [88, 56],
     'renditions': [4, 3],
     'range': [75, 102],
     'fell': [154, 102],
     'chick': [129, 47],
     'style': [470, 604],
     'looked': [485, 250],
     'penis': [22, 6],
     'hippies': [20, 13],
     'white': [423, 506],
     'ethnic': [30, 29],
     'savages': [4, 7],
     'combo': [14, 10],
     'matters': [91, 88],
     'sure': [1008, 884],
     'hispanic': [18, 7],
     'gets': [1102, 1026],
     'grabbed': [16, 19],
     'euro': [22, 9],
     'unnecessary': [164, 69],
     'nasty': [151, 111],
     'costume': [93, 69],
     'park': [112, 112],
     'tribal': [3, 7],
     'around': [1234, 1147],
     'referring': [26, 19],
     'devil': [95, 95],
     'walk': [232, 141],
     'any': [2721, 1874],
     'enjoyed': [274, 640],
     'feel': [911, 1077],
     'storyline': [320, 250],
     'politics': [73, 77],
     'during': [693, 769],
     'given': [681, 596],
     'admittedly': [69, 46],
     'set': [810, 898],
     '1960': [38, 40],
     'footage': [225, 175],
     'war': [376, 549],
     'families': [53, 116],
     'lines': [620, 438],
     'tensions': [7, 20],
     'awakening': [15, 23],
     'unrest': [4, 3],
     'views': [51, 86],
     'original': [1036, 866],
     'drug': [146, 128],
     'dedicated': [32, 52],
     'sexual': [225, 248],
     'basis': [75, 67],
     'news': [125, 115],
     'mixed': [85, 124],
     'sided': [27, 14],
     'succeeded': [46, 41],
     'largely': [78, 91],
     'predictable': [453, 180],
     'fiction': [130, 179],
     'equal': [59, 50],
     'follows': [154, 223],
     'nearly': [306, 298],
     'clips': [63, 46],
     'struggles': [46, 68],
     'consideration': [24, 20],
     'commonplace': [9, 7],
     'historical': [121, 151],
     'promiscuity': [5, 6],
     'racial': [38, 35],
     'figures': [47, 84],
     'vietnam': [52, 52],
     'upheaval': [5, 6],
     'social': [159, 234],
     'aspects': [135, 164],
     'family': [629, 1095],
     'use': [657, 590],
     'fabric': [11, 10],
     'give': [1249, 1134],
     'century': [138, 226],
     'spoke': [45, 29],
     'attempted': [62, 39],
     'whole': [1211, 892],
     'color': [108, 172],
     'half': [860, 500],
     'used': [694, 626],
     'daily': [40, 83],
     'known': [335, 456],
     'lane': [47, 53],
     'stars': [549, 622],
     'tell': [676, 535],
     'laughing': [212, 184],
     'barkin': [6, 5],
     'show': [1274, 1411],
     'diane': [42, 31],
     'trying': [1060, 684],
     'kyle': [15, 22],
     'although': [728, 1014],
     'lost': [513, 543],
     'enjoy': [490, 824],
     'coming': [364, 414],
     'ten': [333, 272],
     'dreyfuss': [11, 11],
     'sister': [226, 263],
     'fifteen': [54, 32],
     'richard': [213, 307],
     'byrne': [11, 13],
     'reason': [1069, 563],
     'fact': [1279, 1130],
     'picked': [142, 113],
     'sight': [119, 117],
     'see': [3123, 3394],
     'actually': [1566, 1161],
     'fixed': [15, 16],
     'gabriel': [19, 21],
     'now': [1443, 1533],
     'fan': [637, 692],
     'mind': [746, 710],
     'goldblum': [19, 23],
     'lately': [36, 37],
     'nothing': [1859, 868],
     'let': [938, 651],
     'being': [2096, 1994],
     '3': [813, 383],
     'going': [1501, 1186],
     'messing': [29, 16],
     'number': [306, 368],
     'him': [1877, 2215],
     'proves': [115, 163],
     'mob': [38, 57],
     'store': [211, 155],
     'consider': [186, 216],
     'pryor': [9, 9],
     'actual': [331, 238],
     'maclachlan': [3, 3],
     'down': [1327, 1173],
     'tries': [534, 355],
     'burt': [45, 46],
     'boss': [98, 148],
     'appearance': [171, 174],
     'll': [1031, 880],
     'terrible': [884, 168],
     'plays': [584, 893],
     'kick': [101, 100],
     'reynolds': [29, 42],
     'parts': [392, 450],
     'mount': [9, 8],
     'save': [541, 197],
     'everyone': [708, 839],
     'suspense': [251, 268],
     'skip': [165, 50],
     'we': [2386, 2421],
     'spoofing': [11, 10],
     'here': [1819, 1641],
     'faris': [11, 3],
     'regina': [13, 1],
     'zucker': [4, 7],
     'developments': [16, 12],
     'why': [1922, 1182],
     'must': [1029, 1150],
     'airplane': [32, 30],
     'disappointing': [245, 62],
     'anderson': [58, 74],
     'before': [1465, 1437],
     'matrix': [66, 48],
     'less': [757, 639],
     'following': [181, 227],
     'previously': [64, 87],
     'gives': [405, 735],
     'invaders': [3, 10],
     'simon': [39, 98],
     'mccarthy': [8, 12],
     'isn': [1175, 873],
     'directs': [30, 57],
     'cindy': [13, 15],
     'globe': [11, 31],
     'again': [1226, 1346],
     'returns': [91, 122],
     'fight': [359, 375],
     'outrageous': [45, 47],
     'taking': [348, 371],
     'conspiracies': [11, 3],
     'begins': [223, 346],
     'anna': [44, 64],
     'moments': [506, 630],
     'funniest': [78, 178],
     'comedy': [857, 1005],
     'world': [879, 1410],
     'reloaded': [9, 1],
     'shows': [565, 936],
     'jenny': [24, 19],
     'denise': [24, 14],
     'keeps': [181, 282],
     'stop': [455, 362],
     'probably': [998, 966],
     'faced': [74, 94],
     'others': [492, 682],
     '13': [94, 75],
     'brenda': [26, 36],
     'miss': [266, 361],
     'alien': [104, 77],
     'hall': [57, 92],
     'queen': [87, 115],
     'mentioned': [224, 214],
     'straight': [342, 258],
     'want': [1448, 1055],
     'strange': [312, 336],
     'quite': [1077, 1345],
     'proportions': [16, 13],
     'either': [838, 499],
     'third': [272, 257],
     'uncovers': [7, 10],
     'hit': [329, 413],
     'overrated': [57, 23],
     'pg': [58, 33],
     'predecessors': [15, 12],
     'main': [810, 708],
     'david': [236, 366],
     'involve': [36, 40],
     'turns': [415, 497],
     'bunch': [410, 170],
     'enjoyable': [205, 426],
     'issue': [98, 114],
     'lame': [460, 62],
     'pamela': [32, 23],
     'sweeps': [5, 15],
     'television': [255, 340],
     'kind': [1009, 832],
     'killer': [449, 311],
     'analyze': [12, 13],
     'signs': [42, 43],
     'freaky': [27, 19],
     'once': [750, 919],
     'ring': [74, 96],
     'ones': [313, 369],
     'threatening': [56, 42],
     'rating': [300, 199],
     'mildly': [100, 30],
     '6': [198, 155],
     'ever': [2006, 1764],
     'getting': [630, 527],
     'jokes': [377, 231],
     'opening': [381, 337],
     'special': [733, 691],
     'unclear': [22, 13],
     'sequence': [264, 313],
     'scary': [383, 236],
     'sets': [292, 345],
     'anthony': [81, 98],
     'sheen': [14, 17],
     'crew': [210, 180],
     'pointless': [335, 36],
     'circles': [26, 17],
     'non': [395, 296],
     'mile': [51, 34],
     'pretty': [1333, 968],
     'reporter': [64, 66],
     'campbell': [20, 31],
     'charlie': [64, 108],
     'funny': [1308, 1185],
     'crop': [10, 15],
     'massive': [94, 68],
     'rest': [746, 587],
     'onslaught': [3, 4],
     'videotapes': [3, 3],
     'including': [336, 444],
     'cameos': [48, 60],
     'soon': [366, 496],
     'rex': [35, 28],
     'daughter': [323, 355],
     '8': [172, 452],
     'richards': [26, 27],
     'underrated': [27, 162],
     'many': [1835, 2269],
     'focus': [159, 209],
     'feelings': [78, 204],
     'source': [68, 87],
     'train': [129, 124],
     'death': [607, 635],
     'malaise': [6, 3],
     'disaster': [158, 71],
     'irritating': [136, 41],
     'perfectly': [115, 363],
     'adroit': [1, 4],
     'supernatural': [74, 69],
     'crystals': [5, 2],
     'cliché': [200, 59],
     'irony': [49, 56],
     'degree': [82, 71],
     'inadvertently': [15, 20],
     'elements': [260, 313],
     'indignation': [4, 1],
     'certain': [237, 308],
     'proud': [50, 91],
     'how': [2689, 2358],
     'between': [934, 1263],
     'sufficiently': [8, 13],
     'implied': [17, 23],
     'burlesque': [7, 8],
     'absent': [40, 26],
     'hopes': [138, 94],
     'almost': [1061, 1057],
     'manipulation': [19, 22],
     'mediocre': [210, 60],
     'wasted': [368, 60],
     'gritty': [31, 117],
     'antichrist': [12, 1],
     'insignificant': [15, 14],
     'force': [171, 205],
     'collective': [25, 20],
     'brilliant': [205, 626],
     'ruined': [113, 50],
     'eerily': [7, 10],
     'sum': [87, 41],
     'recycled': [38, 18],
     'comedic': [94, 148],
     'owner': [63, 131],
     'tapping': [6, 14],
     'study': [77, 104],
     'grotesquely': [5, 3],
     'celluloid': [64, 24],
     'sorry': [413, 153],
     'unbelievably': [71, 20],
     'tragic': [62, 198],
     'trilogy': [47, 93],
     'add': [340, 283],
     'annoying': [531, 143],
     'inconsequential': [18, 7],
     'gruesome': [64, 68],
     'register': [13, 14],
     'though': [1352, 1494],
     'actress': [394, 417],
     'inappropriate': [61, 15],
     'fervently': [4, 2],
     'score': [281, 438],
     'listened': [23, 12],
     'finally': [519, 577],
     'galore': [12, 10],
     'thank': [138, 180],
     'realism': [68, 125],
     'result': [272, 213],
     'torn': [49, 67],
     'our': [647, 847],
     'made': [2726, 2335],
     'angle': [64, 59],
     'dry': [113, 62],
     'refined': [8, 10],
     'laughter': [80, 91],
     'vomiting': [11, 2],
     'fundamentally': [10, 4],
     'mysticism': [9, 14],
     'viewer': [411, 495],
     'new': [1120, 1427],
     'material': [317, 244],
     'goldsmith': [9, 9],
     'imdb': [323, 183],
     'age': [301, 480],
     'inherent': [21, 30],
     'calling': [92, 60],
     'gone': [286, 286],
     'face': [569, 559],
     'normal': [135, 190],
     'induce': [9, 6],
     'may': [939, 1235],
     'believable': [192, 348],
     'incompetence': [23, 6],
     'subject': [216, 260],
     'music': [800, 1021],
     'lucky': [79, 120],
     'transparent': [26, 9],
     'wreck': [70, 24],
     'since': [952, 1047],
     'heard': [406, 409],
     'never': [1985, 2014],
     'behind': [456, 468],
     'subtlety': [41, 38],
     'happened': [419, 344],
     'jaws': [40, 17],
     'stephanie': [19, 17],
     'render': [17, 11],
     'motivating': [3, 6],
     'boorish': [16, 3],
     'slap': [68, 28],
     'three': [717, 740],
     'wall': [127, 126],
     'collection': [112, 159],
     'starving': [18, 5],
     'fill': [117, 61],
     'shark': [42, 20],
     'certainly': [442, 581],
     'painting': [43, 42],
     '1': [907, 376],
     'your': [1916, 1472],
     'gave': [463, 425],
     'always': [779, 1342],
     'having': [902, 877],
     'sale': [25, 17],
     'dialogue': [628, 429],
     'wild': [126, 178],
     'space': [241, 214],
     'exercise': [61, 34],
     'covers': [35, 71],
     'agree': [215, 219],
     'writer': [446, 378],
     'women': [566, 485],
     '90': [251, 128],
     'artists': [53, 83],
     'cleaned': [10, 8],
     'dvd': [654, 838],
     'dental': [8, 13],
     'patience': [34, 31],
     'wonder': [486, 280],
     'teeth': [77, 48],
     'instead': [1019, 520],
     'crack': [50, 32],
     'bought': [169, 160],
     'quality': [479, 434],
     'previous': [232, 227],
     'logic': [122, 59],
     'case': [566, 489],
     'caring': [62, 65],
     'next': [610, 589],
     'asleep': [120, 36],
     'bored': [289, 100],
     'fall': [278, 299],
     'terms': [130, 187],
     'however': [1176, 1174],
     'minus': [39, 16],
     'felt': [567, 467],
     '60': [122, 93],
     'editing': [350, 205],
     'brutal': [88, 140],
     'creative': [111, 150],
     'liked': [414, 597],
     'expect': [453, 433],
     'early': [476, 643],
     'techniques': [40, 60],
     'young': [815, 1365],
     'juvenile': [54, 28],
     'sex': [573, 389],
     'filmmakers': [236, 153],
     'telephone': [10, 8],
     'different': [569, 987],
     'lasts': [27, 24],
     'night': [685, 699],
     'avant': [7, 17],
     'left': [800, 666],
     'problems': [286, 316],
     'ending': [763, 774],
     'experiences': [58, 99],
     'explicit': [26, 63],
     '70': [158, 148],
     'anonymous': [17, 10],
     'tells': [232, 395],
     'laugh': [545, 422],
     'shoot': [228, 113],
     'reveals': [40, 102],
     'ended': [219, 185],
     'anecdote': [3, 6],
     'meets': [156, 334],
     'loud': [196, 150],
     'feels': [292, 288],
     'adds': [81, 182],
     'obscure': [42, 53],
     'pretentious': [150, 43],
     'surrealism': [10, 15],
     'reasons': [220, 210],
     'theirs': [7, 14],
     'switched': [32, 11],
     'serves': [80, 81],
     'own': [894, 1324],
     'disjointed': [63, 14],
     'prospect': [11, 8],
     'fresh': [90, 195],
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

