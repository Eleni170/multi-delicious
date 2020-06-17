# Introduction

The object of study of this project concerns the classification of a set of entries retrieved from the website delicious.com into a list of topics. The dataset is called [DeliciousMIL](https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels). In this dataset, the documents from the webpages were collected, classified into one or more classes from the [labels.txt](https://github.com/Eleni170/multi-delicious/blob/master/Data/labels.txt) file and split to train and test data. Every record is matched to a list with length the number of classes and each item in the list shows whether the record belongs (1) or doesn't belong (0) to class[item]. This information exists at [train-label.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/train-label.dat) for the training set and at [test-label.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/test-label.dat) for the test set.

The files with the 2 sets ([train-data.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/train-data.dat), [test-data.dat](https://github.com/Eleni170/multi-delicious/blob/master/Data/test-data.dat)) have a specific format. At the beginning of every record there is an indication of the number of sentences n in the documents specified at <n> and an indication of the number of words w for every sentence specified at <w>. Moreover, all the words of the documents have been used for the creation of a vocabulary that maps every word to a unique id ([vocabs.txt](https://github.com/Eleni170/multi-delicious/blob/master/Data/vocabs.txt)). An example of this format is :
  
    <2> <8> 6705 5997 8310 3606 674 8058 5044 4836 <4> 4312 5154 8310 4225
    <n> <w> id id id id id id id id <w> id id id id

As an additional step we preprocessed the dataset in order to fit with our needs for training and testing. We investigated 3 different topics using this dataset: learning from Multi-Label data, Multi-Instance learning and class imbalance. In the next section, we present the results of our investigation.

# Text processing

At first we preprocessed the text documents to obtain numerical features. For each word we counted the occurance at each document. The dataset was converted to a 2d matrix with dimensions number_of_documents x number_of_words where each node showed number of times a document i has the word j. Using this convertion, each word is treated as a feature. For this implementation we used ```read_train``` function from [data_processing](https://github.com/Eleni170/multi-delicious/blob/master/processing-text/data_processing.py).

```
  def read_train(path):
    with open(path) as f:
        line = f.readline()
        docs = []
        docs_bin = []
        while line:
            line = line.rstrip('\n').split()
            L = []
            for i in line:
                try:
                    L.append(int(i))
                except:
                    continue
            docs.append(np.array(L))

            words_bin = np.zeros(8520)
            for i in L:
                words_bin[i] += 1

            docs_bin.append(words_bin)
            line = f.readline()
        docs = np.array(docs)
        docs_bin = np.array(docs_bin)
    return docs_bin, docs
```

It is known that in large corpuses some words will be very present that don't carry such meaningful information about the content of the document. Hence, the classifier would treat very well those frequent words and ignore other important tokens for classification. For that reason, we normalized our feature vectors as a next step.

```
def tfIdf(docs_bin):
    tf_transformer = TfidfTransformer(use_idf=True).fit(docs_bin)
    X_train_tf = tf_transformer.transform(docs_bin)
    return X_train_tf
```

Tf–idf term weighting uses the total number of documents in the document set and the number of documents in the document set that contain each term to extract a result.

Once we finished with tfIdf the dataset was ready for document-level multi-label classification. Below is multi-label confusion matrix for each class, using a LinearSVC for training and prediction.

```
Confusion matrix for label programming:
[[2674  332]
 [ 387  590]]
Confusion matrix for label style:
[[3601  154]
 [ 130   98]]
Confusion matrix for label reference:
[[1843  582]
 [ 913  645]]
Confusion matrix for label java:
[[3445  166]
 [ 158  214]]
Confusion matrix for label web:
[[2467  466]
 [ 551  499]]
Confusion matrix for label internet:
[[3189  257]
 [ 405  132]]
Confusion matrix for label culture:
[[3059  222]
 [ 573  129]]
Confusion matrix for label design:
[[2446  458]
 [ 578  501]]
Confusion matrix for label education:
[[2906  274]
 [ 533  270]]
Confusion matrix for label language:
[[3369  131]
 [ 318  165]]
Confusion matrix for label books:
[[3333  143]
 [ 376  131]]
Confusion matrix for label writing:
[[3374  131]
 [ 338  140]]
Confusion matrix for label computer:
[[3243  231]
 [ 367  142]]
Confusion matrix for label english:
[[3514  114]
 [ 243  112]]
Confusion matrix for label politics:
[[3444  147]
 [ 236  156]]
Confusion matrix for label history:
[[3371  171]
 [ 354   87]]
Confusion matrix for label philosophy:
[[3626   88]
 [ 208   61]]
Confusion matrix for label science:
[[3305  177]
 [ 352  149]]
Confusion matrix for label religion:
[[3704   72]
 [ 114   93]]
Confusion matrix for label grammar:
[[3799   51]
 [  95   38]]
```

# Class imbalance

Class imbalance did exist in our dataset. For each class, we counted the number of documents that were subject to the topic and found that some topics were discussed more than others.

| Class | Label Counts |
| ------------- |:-------------:|
|programming | 2050 |
|style       | 479  |
|reference   | 3181 |
|java        | 799  |
|web         | 2203 |
|internet    | 1211 |
|culture     | 1471 |
|design      | 2221 |
|education   | 1559 |
|language    | 1004 |
|books       | 1034 |
|writing     | 939  |
|computer    | 1049 |
|english     | 725  |
|politics    | 830  |
|history     | 898  |
|philosophy  | 598  |
|science     | 1001 |
|religion    | 411  |
|grammar     | 224  |

In the second appoach, we implemented a cost sensitive learning technique, where we gave each record a proper weight. We used a one vs rest technique to find the cost loss for each class. For each record, we created a cost matrix where True Positives and True Negatives were assigned to 0 and False Negatives and False Positives were computed using the following equations:

```
FN= 1/(percentage_of_examples_in class * number_of_samples)
FP = 1/(percentage_of_examples_not_in class * number_of_samples)
```

The reason we used these equations was because the cost of losing a record that belonged to the class we were focusing was greater than classifying a document by mistake to the topic and as a result the percentage of examples that didn't belong to the class was bigger that the percentage that did belong.

| Class | Label Counts |Cost Loss |
| -------------|:-----------:|:-------------:|
|programming | 2050 |0.9982920545807057 |
|style       | 479  |1.0448010839348703 |
|reference   | 3181 |1.015895874965261  |
|java        | 799  |1.0918966241144887 |
|web         | 2203 |1.0088941925218773 |
|internet    | 1211 |1.0171099517860558 |
|culture     | 1471 |1.0206941285880635 |
|design      | 2221 |0.9855036803284339 |
|education   | 1559 |0.9930770616477518 |
|language    | 1004 |0.9874699792960665 |
|books       | 1034 |1.047756608856901  |
|writing     | 939  |0.999238386286178  |
|computer    | 1049 |1.0268828332388904 |
|english     | 725  |1.0102465953382924 |
|politics    | 830  |0.9889704419779607 |
|history     | 898  |0.9694101619567458 |
|philosophy  | 598  |1.0008598030560543 |
|science     | 1001 |0.9360056452287842 |
|religion    | 411  |1.0071414681077542 |
|grammar     | 224  |1.0296787423103213 |

We compared the cost loss to the label counts and we found that the classes that occurred more in the dataset had the smallest cost loss.
