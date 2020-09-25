# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:01:05 2020

@author: Hector
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy.signal import resample
import seaborn as sns

CANVAS_HEIGHT = 400
CANVAS_LENGTH = 800
CANVAS_X_MIN = -80
CANVAS_X_MAX = 80
CANVAS_Y_MIN = -70
CANVAS_Y_MAX = 70
CANVAS_X_MULTIPLIER = CANVAS_LENGTH / (CANVAS_X_MAX - CANVAS_X_MIN)
CANVAS_Y_MULTIPLIER = -CANVAS_HEIGHT / (CANVAS_Y_MAX - CANVAS_Y_MIN)
RESAMPLE_VAL = 60

def decode_number(number):
    switcher = { 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
                 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
                 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S',
                 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}
    return switcher.get(number,str(number))


def characters_merged_37_classes(label):
    """ changes the lower case letter to upper case letters where upper
        and lower letters are similar
    label: ASCII character
        ASCII character a to z

    returns new ASCII character in upper case if letter is detected
    """
    switcher = { 'c': 'C', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M',
                 'o': 'O', 'p': 'P', 's': 'S', 'u': 'U', 'v': 'V', 'w': 'W',
                 'x': 'X', 'y': 'Y', 'z': 'Z' }

    return switcher.get(label,label)


def characters_merged_26_classes(label):
    """ changes the lower case letter to upper case letters
    label: ASCII character
        ASCII character a to z

    returns new ASCII character in upper case if letter is detected
    """
    switcher = { 'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F',
                 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L',
                 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R',
                 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X',
                 'y': 'Y', 'z': 'Z' }

    return switcher.get(label,label)


def print_confusion_matrix(confusion_matrix, class_names, figsize = (30,30),
                           fontsize=14, normalize=True):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print("Confusion matrix, without normalization")

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt= fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def euc_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def euc_dist(x,y):
    dist = []
    for i in range(len(x)):
        if i != 0:
            tmp = np.sqrt(np.square(x[i]-x[i-1]) + np.square(y[i]-y[i-1]))
            dist.append(tmp)
        else:
            dist.append(0.0)
    return np.array(dist)

def get_xy_canvas(data):
    # data : (3,n) np array orientation angles
    # return: (2,n) np array with x & y points
    last_x, last_y = 0, 0
    last_ang_x, last_ang_y = 0, 0
    x = []
    y = []
    for i in range(len(data[0])):
        curr_ang_x = data[1][i]
        curr_ang_y = data[2][i]

        angle_x = (curr_ang_x - last_ang_x) * CANVAS_X_MULTIPLIER
        angle_y = (last_ang_y - curr_ang_y) * CANVAS_Y_MULTIPLIER
        pan_x = np.around(angle_x + last_x)
        pan_y = np.around(angle_y + last_y)
        last_ang_x, last_ang_y = curr_ang_x, curr_ang_y
        last_x, last_y = pan_x, pan_y
        x.append(pan_x)
        y.append(pan_y)
    return np.array([x,y])

def get_angles_offset(data):
    # data : (3,n) np array orientation angles
    # return: (3,n) np array with roll, yaw, pitch angles offset to start point
    last_yaw, last_pitch, last_roll = 0, 0, 0
    last_ang_x, last_ang_y, last_ang_z = 0, 0, 0
    off_x, off_y, off_z = 0, 0 , 0
    yaw = []
    pitch = []
    roll = []
    for i in range(len(data[0])):
        curr_ang_x = data[1][i] - off_x
        curr_ang_y = data[2][i] - off_y
        curr_ang_z = data[0][i] - off_z

        if i == 0:
            off_x = curr_ang_x
            off_y = curr_ang_y
            off_z = curr_ang_z

        angle_x = curr_ang_x - last_ang_x
        angle_y = curr_ang_y - last_ang_y
        angle_z = curr_ang_z - last_ang_z

        yaw_offset = angle_x + last_yaw
        pitch_offset = angle_y + last_pitch
        roll_offset = angle_z + last_roll

        last_ang_x, last_ang_y, last_ang_z = curr_ang_x, curr_ang_y, curr_ang_z
        last_yaw, last_pitch, last_roll = yaw_offset, pitch_offset, roll_offset
        if i == 0:
            yaw.append(0.0)
            pitch.append(0.0)
            roll.append(0.0)
        else:
            yaw.append(yaw_offset)
            pitch.append(pitch_offset)
            roll.append(roll_offset)
    return np.array([roll, yaw, pitch])

def f_features(dataframe):
    # Data used to get features
    #gyr = dataframe.iloc[:,9:12].values.transpose()
    #lin = dataframe.iloc[:,12:15].values.transpose()
    eul = dataframe.iloc[:,15:18].values.transpose()

    # calculate features
    #xy = get_xy_canvas(eul)
    #eul = get_angles_offset(eul)
    #xy_euc = euc_dist(xy[0],xy[1])

    # Features
    # feat_1 : gyr_x
    # feat_2 : gyr_y
    # feat_3 : gyr_z
    # feat_4 : lin_x
    # feat_5 : lin_y
    # feat_6 : lin_z
    # feat_7 : eul_roll
    # feat_8 : eul_yaw
    # feat_9 : eul_pitch
    # feat_10 : x_pos
    # feat_11 : y_pos
    # feat_12 : euc_dist_xy

    # Add features to feature list
    #features = [gyr[0], gyr[1], gyr[2], lin[0], lin[1], lin[2], eul[0], eul[1],
    #            eul[2], xy[0], xy[1], xy_euc]
    #features = [lin[0], lin[1], lin[2]]
    features = [eul[0], eul[1], eul[2]]
    features = np.array(features).transpose()

    # Feature scaling
    #sc = StandardScaler()
    #features = sc.fit_transform(features)

    # Feature scaling maximum absolute value
    ma = MaxAbsScaler()
    features = ma.fit_transform(features)

    # Resampling to 60 (Median lenght in database)
    features = resample(features, RESAMPLE_VAL)

    return features

def f_features2(dataframe):
    # Data used to get features
    eul = dataframe.iloc[:,15:18].values.transpose()

    # calculate features
    xy = get_xy_canvas(eul)

    # Add features to feature list
    features = [xy[0],xy[1]]
    features = np.array(features).transpose()

    # Feature scaling standard
    #sc = StandardScaler()
    #features = sc.fit_transform(features)

    # Feature scaling maximum absolute value
    ma = MaxAbsScaler()
    features = ma.fit_transform(features)

    # Resampling to 60 (Median lenght in database)
    features = resample(features, RESAMPLE_VAL)

    return features


