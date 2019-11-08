from __future__ import division
import pandas
from collections import OrderedDict
import itertools
from numpy import *
import numpy as np
from sklearn.linear_model import LogisticRegression

#This method is used to read the input file
# line by line and stores it in a variable.
def readFile(filePath):
    fileInput = ""
    f = open(filePath, "r")
    fileInput = fileInput + f.read()
    return fileInput


#This method is used to separate text from their labels and also make separate dataset for 0 labels and 1 labels.
# It returns three dataframes of pandas.
# The dataFrame consists of all the sentences and their labels, dataFrame\_0 consists of all the texts and labels
# for the label 0 and dataFrame\_1 consists of all the texts and labels for the label 1.

def dataPreProcessing(data):
    fileInput = dataPrep(data)
    labels, texts, labels_0, texts_0, labels_1, texts_1  = [], [], [], [], [], []
    for i, line in enumerate(fileInput.split("\n")):
        content = [word.lower() for word in line.split()] #line.split() #
        if(len(content)!= 0):
            labels.append(content[-1])
            texts.append(" ".join(content[0:-1]))
            if content[-1] == '0':
                labels_0.append(content[-1])
                texts_0.append(" ".join(content[0:-1]))
            else:
                labels_1.append(content[-1])
                texts_1.append(" ".join(content[0:-1]))
    # create a dataframe using texts and lables
    dataFrame = pandas.DataFrame()
    dataFrame_0 = pandas.DataFrame()
    dataFrame_1 = pandas.DataFrame()
    dataFrame['text'] = texts
    dataFrame['label'] = labels
    dataFrame_0['text'] = texts_0
    dataFrame_0['label'] = labels_0
    dataFrame_1['text'] = texts_1
    dataFrame_1['label'] = labels_1
    return dataFrame, dataFrame_0, dataFrame_1

#This method handles the punctuation in the corpus.
# To get the correct unique words or vocabulary from the dataset.
def dataPrep(data):
    punctuations_1 = ''''"\<>@#$%^&_~'''
    punctuations_2 = ",.!?+=*-():;{}[]/"
    text = ""
    for char in data:
        if char in punctuations_2:
            text = text + " "

        elif char not in punctuations_1:
            text = text + char

    return text

#This method takes in the sentence and using the start index and end index
# returns words by sliding the window over the sentence.
def window(startIndex, endIndex, corpus):
    word = ""
    for i in range(startIndex, endIndex):
        if i == endIndex-1:
            word +=corpus[i]
        else:
            word += corpus[i]+" "
    return word

#This method returns the vocabulary for a specified n i.e. for n = 1 returns the vocabulary for f1,
#  n=2 returns vocabulary for f2 and so on
def createVocab(df, n):
    dict = OrderedDict()
    for sentence in df['text']:
        text = sentence.split()
        for i in range(len(text)):
            startIndex = i;
            endIndex = i + n;
            if endIndex <= len(text):
                word = window(startIndex, endIndex, text)
                if dict.has_key(word):
                    dict[word] += 1
                else:
                    dict[word] = 1

    vocabulary = dict.keys()
    return vocabulary

#Fetches the column number to be updated in an array.
def getColumnNumber(wordMatch, fv):
    # for i in range(len(fv)):
    #     if wordMatch == fv[i]:
    #         return i
    i = fv.index(wordMatch)
    return i


#This methods create the feature vector representation for unigrams.
#  It takes the dataframe, the vocabulary for unigram and returns a 2d array
# with feature vector representation of unigrams.
def createngrams(df, vocabulary, text):
    for d in df:
        new = []
        sentence = d.split()
        for vocab in vocabulary:
            for word in sentence:
                if word == vocab:
                    if not new:
                        for ini in range(len(vocabulary)):
                            new.append(0)
                    columnNumber = getColumnNumber(word, vocabulary)
                    new[columnNumber]+=1
        text.append(new)

    return text

#This methods create the feature vector representation for n=2 to 5.
# It takes the dataframe, the vocabulary for the respective n and
# returns a 2d array with feature vector representation of it.
def generatengrams(df, vocabulary, text):
    for d in df:
        new = []
        for vocab in vocabulary:
            if not new:
                new = [0] * len(vocabulary)
            count = d.count(vocab)
            columnNumber = getColumnNumber(vocab, vocabulary)
            new[columnNumber] = count
        text.append(new)

    return text


def createDataset(dataFrame, columnFeature, columnLabel):
    dataset = []

    dataset.append((dataFrame[columnFeature].values, dataFrame[columnLabel].values))
    return dataset

#extract the required data from the dataframe for training and testing
def getdataframedata(dataset, columnName):
    features, labels = [], []
    fold_training_set = list(itertools.chain.from_iterable(dataset))
    for i in range(len(fold_training_set)):
        features.append(fold_training_set[i][columnName].tolist())
        labels.append(fold_training_set[i]['label'].tolist())
    f = list(itertools.chain.from_iterable(features))
    l = list(itertools.chain.from_iterable(labels))
    return f,l

#extract the required data from the dataframe for training and testing
def gettestdataframedata(dataset, columnName):
    features, labels = [], []
    for i in range(len(dataset)):
        features.append(dataset[i][columnName].tolist())
        labels.append(dataset[i]['label'].tolist())
    f = list(itertools.chain.from_iterable(features))
    l = list(itertools.chain.from_iterable(labels))
    return f,l

#extract the required data from the dataframe for training and testing
def getFeaturesData(train_set, test_set, feature):
    f1_train_feature, f1_train_label = getdataframedata(train_set, feature)
    f1_test_feature, f1_test_label = gettestdataframedata(test_set, feature)
    return f1_train_feature, f1_train_label, f1_test_feature, f1_test_label


#function used to evaluate prediction and calculate accuracy
def evaluate_prediction(predictions, answers):
    correct = sum(asarray(predictions) == asarray(answers))
    total = float(prod(len(answers)))
    return correct / total

#This method creates a confusion matrix based on the predictions and actual labels and
#  returns true\_positive, false\_positive, true\_negative, false\_negative
def confusionmatrix(predictions, actuallabels):
    row = {}
    for i in range(len(actuallabels)):
        x = str(actuallabels[i]) + str(predictions[i])
        key = "group_{0}".format(x)
        if key in row:
            row["group_{0}".format(x)] = row["group_{0}".format(x)] + 1
        else:
            row["group_{0}".format(x)] = 1

    labelrows = []
    for x in range(0, 2):
        for y in range(0, 2):
            j = str(x) + str(y)
            p = "group_{0}".format(j)
            if p in row:
                labelrows.append(row["group_{0}".format(j)])
            else:
                labelrows.append(0)

    cm = reshape(labelrows, (2, 2))
    true_positive = cm[1][1]
    false_positive = cm[0][1]
    true_negative = cm[0][0]
    false_negative = cm[1][0]
    return true_positive, false_positive, true_negative, false_negative


#This method calculates the metrics precision, recall, true positive rate,
# false positive rate, true negative rate, false negative rate.
def calculateMetrix(predictions, actuallabels):
    true_positive, false_positive, true_negative, false_negative = confusionmatrix(predictions, actuallabels)

    precision = round(float(true_positive / (true_positive + false_positive)), 5)

    recall = round(float(true_positive / (true_positive + false_negative)), 5)

    tpr = round(float(true_positive / (true_positive + false_negative)), 5)

    fpr = round(float(false_positive / (true_negative + false_positive)), 5)

    tnr = round(float(true_negative / (true_negative + false_positive)), 5)

    fnr = round(float(false_negative / (true_positive + false_negative)), 5)

    return (precision, recall, tpr, fpr, tnr, fnr)


#This method contains the core code for n-gram feature representation,
# 10-fold cross validation, binary classification.
def ngrammodelclassification(filePath):
    print("*********************Starting with dataset************************** :", filePath)
    fileInput = readFile(filePath)
    dataFrame, dataFrame_0, dataFrame_1 = dataPreProcessing(fileInput)

    text_f1, text_f2, text_f3, text_f4, text_f5 = [], [], [], [], []

    # for f1
    vocab_f1 = createVocab(dataFrame, 1)
    text_f1 = createngrams(dataFrame_0['text'], vocab_f1, text_f1)
    dataFrame_0['text_f1'] = text_f1

    text_f1 = []
    text_f1 = createngrams(dataFrame_1['text'], vocab_f1, text_f1)

    dataFrame_1['text_f1'] = text_f1

    print("f1 done")

    # for f2
    vocab_f2 = createVocab(dataFrame, 2)
    text_f2 = generatengrams(dataFrame_0['text'], vocab_f2, text_f2)
    dataFrame_0['text_f2'] = text_f2

    text_f2 = []
    text_f2 = generatengrams(dataFrame_1['text'], vocab_f2, text_f2)
    dataFrame_1['text_f2'] = text_f2

    print("f2 done")

    # for f3
    vocab_f3 = createVocab(dataFrame, 3)
    text_f3 = generatengrams(dataFrame_0['text'], vocab_f3, text_f3)
    dataFrame_0['text_f3'] = text_f3

    text_f3 = []
    text_f3 = generatengrams(dataFrame_1['text'], vocab_f3, text_f3)
    dataFrame_1['text_f3'] = text_f3

    print("f3 done")

    # for f4
    vocab_f4 = createVocab(dataFrame, 4)

    text_f4 = generatengrams(dataFrame_0['text'], vocab_f4, text_f4)
    dataFrame_0['text_f4'] = text_f4

    text_f4 = []
    text_f4 = generatengrams(dataFrame_1['text'], vocab_f4, text_f4)
    dataFrame_1['text_f4'] = text_f4

    print("f4 done")

    # for f5
    vocab_f5 = createVocab(dataFrame, 5)
    text_f5 = []
    text_f5 = generatengrams(dataFrame_0['text'], vocab_f5, text_f5)
    dataFrame_0['text_f5'] = text_f5

    text_f5 = []
    text_f5 = generatengrams(dataFrame_1['text'], vocab_f5, text_f5)
    dataFrame_1['text_f5'] = text_f5

    print("f5 done")

    # create combined features
    dataFrame_0['text_f1f2'] = dataFrame_0['text_f1'] + dataFrame_0['text_f2']
    dataFrame_1['text_f1f2'] = dataFrame_1['text_f1'] + dataFrame_1['text_f2']

    dataFrame_0['text_f1f2f3'] = dataFrame_0['text_f1'] + dataFrame_0['text_f2'] + dataFrame_0['text_f3']
    dataFrame_1['text_f1f2f3'] = dataFrame_1['text_f1'] + dataFrame_1['text_f2'] + dataFrame_1['text_f3']

    dataFrame_0['text_f1f2f3f4'] = dataFrame_0['text_f1'] + dataFrame_0['text_f2'] + dataFrame_0['text_f3'] + \
                                   dataFrame_1['text_f4']
    dataFrame_1['text_f1f2f3f4'] = dataFrame_1['text_f1'] + dataFrame_1['text_f2'] + dataFrame_1['text_f3'] + \
                                   dataFrame_1['text_f4']

    dataFrame_0['text_f1f2f3f4f5'] = dataFrame_0['text_f1'] + dataFrame_0['text_f2'] + dataFrame_0['text_f3'] + \
                                     dataFrame_0['text_f4'] + dataFrame_0['text_f5']
    dataFrame_1['text_f1f2f3f4f5'] = dataFrame_1['text_f1'] + dataFrame_1['text_f2'] + dataFrame_1['text_f3'] + \
                                     dataFrame_1['text_f4'] + dataFrame_1['text_f5']

    dataset_0 = np.array_split(dataFrame_0, 10)
    dataset_1 = np.array_split(dataFrame_1, 10)

    all_fold_all_feature_score, all_fold_all_feature_precision, all_fold_all_feature_recall, all_fold_all_feature_tpr, all_fold_all_feature_fpr, all_fold_all_feature_tnr, all_fold_all_feature_fnr = [], [], [], [], [], [], []
    features = ["text_f1", "text_f2", "text_f3", "text_f4", "text_f5", "text_f1f2", "text_f1f2f3", "text_f1f2f3f4",
                "text_f1f2f3f4f5"]
    for feature in features:
        score, precision_feature, recall_feature, tpr_feature, fpr_feature, tnr_feature, fnr_feature = [], [], [], [], [], [], []
        for fold in range(len(dataset_0)):
            training_set_0 = []
            training_set_0 = dataset_0[:]
            test_labels = []
            testing_set_0 = []
            train_set_fold_copy_0 = training_set_0[fold]
            test_set_0 = train_set_fold_copy_0[:]
            del training_set_0[fold]

            training_set_1 = []
            training_set_1 = dataset_1[:]
            testing_set_1 = []
            train_set_fold_copy_1 = training_set_1[fold]
            test_set_1 = train_set_fold_copy_1[:]
            del training_set_1[fold]

            train_set = []
            test_set = []
            train_set.append(training_set_0)
            train_set.append(training_set_1)

            test_set.append(test_set_0)
            test_set.append(test_set_1)

            # Fold feature dataset

            train_feature, train_label, test_feature, test_label = getFeaturesData(train_set, test_set, feature)

            logisticRegr = LogisticRegression()
            logisticRegr.fit(train_feature, train_label)
            predictions = logisticRegr.predict(test_feature)
            # score = logisticRegr.score(test_feature, test_label)
            accuracy = evaluate_prediction(predictions.tolist(), test_label)
            score.append(accuracy)
            precision, recall, tpr, fpr, tnr, fnr = calculateMetrix(predictions.tolist(), test_label)
            precision_feature.append(precision)
            recall_feature.append(recall)
            tpr_feature.append(tpr)
            fpr_feature.append(fpr)
            tnr_feature.append(tnr)
            fnr_feature.append(fnr)

        print("score per fold for  feature :", feature, "  is :", score)
        print("precision per fold for feature :", feature, "  is :", precision_feature)
        print("recall per fold for feature :", feature, "  is :", recall_feature)
        print("tpr per fold for feature :", feature, "  is :", tpr_feature)
        print("fpr per fold for feature :", feature, "  is :", fpr_feature)
        print("tnr per fold for feature :", feature, "  is :", tnr_feature)
        print("fnr per fold for feature :", feature, "  is :", fnr_feature)
        average_score = sum(score) / 10
        average_precision = sum(precision_feature) / 10
        average_recall = sum(recall_feature) / 10
        average_tpr = sum(tpr_feature) / 10
        average_fpr = sum(fpr_feature) / 10
        average_tnr = sum(tnr_feature) / 10
        average_fnr = sum(fnr_feature) / 10

        all_fold_all_feature_score.append((feature, round(float(average_score*100),5)))
        all_fold_all_feature_precision.append((feature, average_precision))
        all_fold_all_feature_recall.append((feature, average_recall))
        all_fold_all_feature_tpr.append((feature, average_tpr))
        all_fold_all_feature_fpr.append((feature, average_fpr))
        all_fold_all_feature_tnr.append((feature, average_tnr))
        all_fold_all_feature_fnr.append((feature, average_fnr))

    print("all_fold_all_feature_score :", all_fold_all_feature_score)
    print("all_fold_all_feature_precision :", all_fold_all_feature_precision)
    print("all_fold_all_feature_recall :", all_fold_all_feature_recall)
    print("all_fold_all_feature_tpr :", all_fold_all_feature_tpr)
    print("all_fold_all_feature_fpr :", all_fold_all_feature_fpr)
    print("all_fold_all_feature_tnr :", all_fold_all_feature_tnr)
    print("all_fold_all_feature_fnr :", all_fold_all_feature_fnr)
    print("******************End of dataset*********************** :", filePath)

filePath_1 = ("./dataset/amazon_cells_labelled.txt")
ngrammodelclassification(filePath_1)

filePath_2 = ("./dataset/imdb_labelled.txt")
ngrammodelclassification(filePath_2)

filePath_3 = ("./dataset/yelp_labelled.txt")
ngrammodelclassification(filePath_3)

