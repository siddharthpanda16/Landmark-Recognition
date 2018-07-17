# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:07:33 2018

@author: siddh
"""

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#import argparse
#import utils
import cv2
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import urllib
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing


descriptorsValues_all = []
descriptorsValues_all_test = []
labels_all = []
labels_all_test = []
winSize = (64,32)
blockSize = (8,8)
blockStride = (2,2)
cellSize = (2,2)
nbins = 9
pca = PCA(.97)
results = []
names = []

#Test
def test(clf,X_test,y_test):
   print("Testing...")
   for i, url in enumerate(X_test):
      test_img = urlToImage(url)
      if not test_img is None:
         test_img = cv2.resize(test_img, (64,32) );  
         test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY);

         test_d = cv2.HOGDescriptor( winSize,blockSize,blockStride,cellSize,nbins );
         test_descriptorsValues = test_d.compute(test_img_gray);
         test_descriptorsValues = test_descriptorsValues.T
         descriptorsValues_all_test.append(test_descriptorsValues)
         labels_all_test.append(y_test.values[i])

   descriptorsValues_all_test_arr = np.array(descriptorsValues_all_test)
   dataset_size = len(descriptorsValues_all_test_arr)
   descriptorsValues_all_test_arr = descriptorsValues_all_test_arr.reshape(dataset_size,-1)
   descriptorsValues_all_test_arr = pca.transform(descriptorsValues_all_test_arr)   
   pred = clf.best_estimator_.predict(descriptorsValues_all_test_arr)
   accuracy = accuracy_score(labels_all_test, pred)
   print("Accuracy : " , accuracy)
   results.append(accuracy)
   print(classification_report(labels_all_test, pred))
   #TO-DO : print roc curve here
   print('area under ROC curve')
   lb = preprocessing.LabelBinarizer()
   y_test_b=lb.fit_transform(labels_all_test)
   probs = clf.predict_proba(descriptorsValues_all_test_arr)
   preds = probs[:,1]
   y_test_bs = y_test_b[:,1]
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_bs, preds)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   
   print(roc_auc)
   print('\n')
   print('RoC curve')
   #TO-DO : Roc Plot here
   
   plt.title('Receiver Operating Characteristic')
   plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f'% roc_auc)
   plt.legend(loc='lower right')
   plt.plot([0,1],[0,1],'r--')
   plt.xlim([-0.1,1.2])
   plt.ylim([-0.1,1.2])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.show()

def urlToImage(url):
   try:
      # download the image
      resp = urllib.request.urlopen(url)
      #convert it to a NumPy array
      image = np.asarray(bytearray(resp.read()), dtype="uint8")
      #read it into OpenCV format
      image = cv2.imdecode(image, cv2.IMREAD_COLOR)
      return image
   except HTTPError as e:
      print()
   except URLError as e:
      print()

def trainTestModel(clf,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test):
   #clf=SVC(C=10000,kernel="linear")
   #clf.fit(descriptorsValues_all_arr,labels_all,sample_weight=None)
   classifier = GridSearchCV(estimator=clf, param_grid=parameters)
   classifier.fit(descriptorsValues_all_arr,labels_all)
   print("Best estimator : " , classifier.best_estimator_)
   test(classifier,X_test,y_test)

def trainTestSVM(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training SVM...")
   parameters = {'C':[1.0,2.0,5.0], 'kernel' : ["linear", "rbf", "poly"], 'degree' : [3,4,5],'gamma':[1.0,1.5,2.0], 'max_iter':[-1],'random_state' : [1,2,3]}
   svc = SVC(probability=True)
   names.append("SVM");
   trainTestModel(svc,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)

def trainTestMLP(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training MLP...")
   mlp = MLPClassifier()
   names.append("MLP");
   parameters = {'hidden_layer_sizes':[70,100,120], 'activation' : ["logistic", "tanh", "relu"], 'learning_rate' : ["constant", "invscaling", "adaptive"], 'max_iter':[750,1000]}
   trainTestModel(mlp,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)

def trainTestDT(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training DT...")
   dt = DecisionTreeClassifier()
   names.append("DT");
   parameters = {'max_depth':[35,50,70,90], 'max_features' : [4,"sqrt","log2"], 'max_leaf_nodes' : [35, 50, 25, 10],'min_samples_split':[5,3,7]}
   trainTestModel(dt,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)


def trainTestLR(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training LR...")
   lr = linear_model.LogisticRegression()
   names.append("LR");
   parameters = {'C':[1,5,10], 'max_iter' : [10,20,50], 'fit_intercept' : [True, False], 'penalty':["l1","l2"]}
   trainTestModel(lr,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)

def trainTestGNB(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training GNB...")
   gb = GaussianNB()
   names.append("GNB");
   parameters = {'priors':[None]}  
   trainTestModel(gb,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)


def trainTestKNN(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training KNN...")
   knn = KNeighborsClassifier()
   names.append("KNN");
   parameters = {"n_neighbors": [5,6,7], "algorithm": ['auto','brute'],"p": [2,3],"weights": ['uniform','distance']}   
   trainTestModel(knn,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)

def trainTestAdaBoost(descriptorsValues_all_arr, labels_all, X_test, y_test):
   print("Training AdaBoost...")
   ada = AdaBoostClassifier()
   names.append("AdaBoost");
   parameters = {"n_estimators": [80], "learning_rate": [0.8],"random_state": [80],"algorithm": ['SAMME']}
   trainTestModel(ada,parameters,descriptorsValues_all_arr, labels_all, X_test, y_test)

def main():
   dataset = pd.read_csv('train_01.csv')
   urls = dataset['url']
   ids = dataset['landmark_id']

   X_train, X_test, y_train, y_test = train_test_split( urls, ids, test_size=0.2, random_state=1)
   print("Downloading images...")
   for i, url in enumerate(X_train):
      # download the image URL and display it
      img = urlToImage(url)
      if not img is None:
         #print("Downloading ", i , "...")
         img = cv2.resize(img, (64,32) );  
         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);

         d = cv2.HOGDescriptor( winSize,blockSize,blockStride,cellSize,nbins );
         locations = ((1, 2), )
         descriptorsValues = d.compute( img_gray );
         descriptorsValues = descriptorsValues.T

         descriptorsValues_all.append(descriptorsValues)
         labels_all.append(y_train.values[i])
      
   #print(len(descriptorsValues_all))
   #print(len(labels_all))

   #convert the list into numpy array and flatten the 3d array to 2d array
   descriptorsValues_all_arr = np.array(descriptorsValues_all)
   dataset_size = len(descriptorsValues_all_arr)
   descriptorsValues_all_arr = descriptorsValues_all_arr.reshape(dataset_size,-1)
   #print(descriptorsValues_all_arr.shape)

   print("Performing principal component analysis...")
   pca.fit(descriptorsValues_all_arr)
   #print(pca.n_components_)
   descriptorsValues_all_arr = pca.transform(descriptorsValues_all_arr)

   trainTestSVM(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestGNB(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestKNN(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestLR(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestDT(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestMLP(descriptorsValues_all_arr, labels_all, X_test, y_test)
   trainTestAdaBoost(descriptorsValues_all_arr, labels_all, X_test, y_test)


   # boxplot algorithm comparison
   plot = plt.figure()
   plot.suptitle('Algorithm Comparison')
   axes = plot.add_subplot(111)
   plt.boxplot(results)
   axes.set_xticklabels(names)
   plt.show()


main()