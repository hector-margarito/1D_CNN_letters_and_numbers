# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:37:38 2020

@author: Hector
# Importing the libraries
"""
import numpy as np
import matplotlib.pyplot as plt
import feature_extraction as fex
from sklearn.metrics import classification_report, confusion_matrix
import OS_CNN as oscnn
from Classifiers.OS_CNN.OS_CNN_easy_use import OS_CNN_easy_use
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

Result_log_folder = './data/online_data/prepared/1D_CNN_36_classes_model/'
dataset_name = "AirFinger_2"

X_train = np.load(Result_log_folder + "X_train.npy")
X_test = np.load(Result_log_folder + "X_test.npy")
y_train = np.load(Result_log_folder + "y_train.npy")
y_test = np.load(Result_log_folder + "y_test.npy")

X_train, X_validate, y_train , y_validate  = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0)

# creat model and log save place
model = oscnn.OS_CNN_easy_use(
        Result_log_folder = Result_log_folder, # the Result_log_folder
        dataset_name = dataset_name,           # dataset_name for creat log under Result_log_folder
        device = "cuda:0",                     # Gpu
        max_epoch = 100,                        # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough
        paramenter_number_of_layer_list = [8*128*X_train.shape[1], 5*128*256 + 2*256*128],
        batch_size=16,
        print_result_every_x_epoch = 1,
        learning_rate=0.005
        )

model.fit(X_train, y_train, X_validate, y_validate)
y_predict = model.predict(X_test)

y_predict_f_name = Result_log_folder + dataset_name + "/y_predict.npy"
np.save(y_predict_f_name,y_predict)

print('correct:',y_test)
print('predict:',y_predict)
acc = accuracy_score(y_predict, y_test)
print(acc)
conf_matrix = confusion_matrix(y_test, y_predict)
#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
#                   'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
#                   'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n',
#                   'q', 'r', 't']
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
               'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X', 'Y', 'Z']
fex.print_confusion_matrix(conf_matrix, class_names)