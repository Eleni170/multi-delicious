# Introduction

The object of study of this project concerns the classification of a set of entries retrieved from the website delicious.com into a list of topics. The dataset is called [DeliciousMIL](https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels). In this dataset, the documents from the webpages were collected, classified into one or more classes from the [labels.txt](https://github.com/Eleni170/multi-delicious/blob/master/Data/labels.txt) file and split to train and test data. Every record is matched to a list with length the number of classes and each item in the list shows whether the record belongs (1) or doesn't belong (0) to class[item]. This information exists at [train-label.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/train-label.dat) for the training set and at [test-label.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/test-label.dat) for the test set.

The files with the 2 sets ([train-data.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/train-data.dat), [test-data.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/test-data.dat)) have a specific format. At the beginning of every record there is an indication of the number of sentences n in the documents specified at <n> and an indication of the number of words w for every sentence specified at <w>. Moreover, all the words of the documents have been used for the creation of a vocabulary that maps every word to a unique id ([vocabs.txt](https://github.com/Eleni170/multi-delicious/blob/master/Data/vocabs.txt)). An example of this format is :
  
    <2> <8> 6705 5997 8310 3606 674 8058 5044 4836 <4> 4312 5154 8310 4225
    <n> <w> id id id id id id id id <w> id id id id

As an additional step we preprocessed the dataset in order to fit with our needs for training and testing. We investigated 3 different topics using this dataset: learning from Multi-Label data, Multi-Instance learning and class imbalance. Below are the results of our investigation.

# Text processing

# Learning from Multi-Label data

# Multi-Instance learning

# Class imbalance
