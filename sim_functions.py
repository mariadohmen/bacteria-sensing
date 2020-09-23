# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:20:00 2020

@author: Maria
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from warnings import warn


def response_sensor_set(n_sens, n_bac, response_limits, indist):
    """
     .

    Parameters
    ----------
    n_sens : int
        Number of sensors in the simulated experiment.
    n_bac : int
        Number of bactera species in simulated experiement.
    response_limit : tuple
        minimal and maximal sensor response.
    indist : float
        float number in an interval [0.0, 1.0]. Percentage of sensors which
        respond the same to two different bacteria.

    Returns
    -------
    response_matix : 2D array
        Array of the true (random) responses for a given numer sensors and
        bacteria cultures.
    """
    response_matrix = np.zeros((n_sens, n_bac))

    # Define the ranges for a positive or negative response
    response_range = np.array([[response_limits[0], 1],
                               [1, response_limits[1]]])

    # radomly decide whether the bacterium gives a positive or negative
    # response
    response_type = (np.random.random(n_bac) < 1/2.).astype(int)

    # Fill the array with random numbers according to the response type
    for i in np.arange(n_sens):
        for j in np.arange(n_bac):
            response_matrix[i, j] = random.uniform(
                    *response_range[response_type[j]])

    # set a subset of responses as the same
    for j in np.arange(1, n_bac):
        indist_mask = (np.random.random(n_sens) < float(indist)).astype(bool)
        if response_type[j] == response_type[j-1]:
            response_matrix[:, j][
                    indist_mask] = response_matrix[:, j-1][indist_mask]
        else:
            response_matrix[:, j][indist_mask] = 1
            response_matrix[:, j-1][indist_mask] = 1

    return response_matrix


def add_noise(response_matrix, noise):
    """Adds random noise to a set of sensor responses.

    Parameters
    ----------
     response_matix : 2D array
        Array of the true (random) responses for a given numer sensors and
        bacteria cultures.
    noise : float
        Random noise level to be added to a set of responses.

    Returns
    -------
    response_matrix : 2D array
        Array of the noisy responses for a given numer sensors and
        bacteria cultures.
    """
    I, J = response_matrix.shape
    # The size of the noise is varried

    noise_size = np.zeros((I, J))
    for i in np.arange(I):
        for j in np.arange(J):
            noise_size[i, j] = random.uniform(-noise, noise)
    # Noise is added/subtracted from response matrix
    response_matrix = response_matrix + response_matrix*noise_size
    return response_matrix


def experiment_repetition(n_sens, n_bac, response_limits, indist, repeats,
                          noise):
    """
    Simulates repetitions of an experiment.

    Parameters
    ----------
    n_sens : int
        Number of sensors in the simulated experiment.
    n_bac : int
        Number of bactera species in simulated experiement.
    response_limit : tuple
        minimal and maximal sensor response.
    indist : float
        float number in an interval [0.0, 1.0]. Percentage of sensors which
        respond the same to two different bacteria.
    repeats : int
        Number of experiment repetitions.
    noise : float
        Random noise level to be added to a set of responses.

      Returns
    -------
    response_cube : 23 array
        Array of the noisy responses for a given numer sensors and
        bacteria cultures repeated a certain number of times.
    """
    response_cube = np.zeros((n_sens, n_bac, repeats))

    # Randomly determine the responses for a sensor and bacteria set
    response_matrix = response_sensor_set(
            n_sens, n_bac, response_limits, indist)

    # simulate repetition of the same experiment by adding noise
    for r in np.arange(repeats):
        response_cube[:, :, r] = add_noise(response_matrix, noise)
    return response_cube


def simple_PCA(dataset, n_sens, figure=False):
    """"
    Calculates a simple PCA.

    Parameters
    ----------
    dataset : DataFrame
        Pandas DataFrame, where the columns are the response of each sensor.
        The last column should be called bacterium number and contains a
        numerical key for the bacterium.
    n_sens : int
        Number of sensors used in measurement/simulation. Equal to the number
        of principal components.
    figure : bool, optional
        If figure is true, a figure plotting the first two principal components
        is returned. Default : False

    Returns
    -------
    explained_variance : lst
        List of the variance ratio of the principal components of the PCA.
    x_pca : 2D array
        Principal components of the simultated data.
    fig : figure
        Matplotlib figure of the first two principal components colored by
        bacterium number.

    """
    scalar = StandardScaler()

    # fitting
    scalar.fit(dataset.iloc[:, :-1])
    scaled_data = scalar.transform(dataset.iloc[:, :-1])

    # Components equals the number of sensors
    pca = PCA(n_components=n_sens)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_

    if figure is False:
        return explained_variance, x_pca

    # plotting results
    fig = plt.figure(figsize=(8, 6))

    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=dataset['bacterium number'],
                cmap='plasma')

    # labeling x and y axes
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    return explained_variance, fig


def perform_LDA(dataset, n_sens):
    """
    PCA which uses a training set for prediction of additional data: trance of
    confusion matrix can be used for judging the quality of assignment.

    Parameters
    ----------
    dataset : DataFrame
        Pandas DataFrame, where the columns are the response of each Sensor.
        The last column should be called bacterium number and contains a
        numerical key for the bacterium.
    n_sens : int
        Number of sensors used in measurement/simulation. Equal to the number
        of principal components.

    Returns
    ------
    fig : Figure
        Matplotlib figure showing the borders for the assignment of the
        logistic regression of the test case and the correct classification in
        the legend.
    cm : 2D array
        Confusion matrix. The diagonal elements were correctly assigned to
        their respective bacterium group.
    """
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2,
                                                        random_state=0)

    # Applying PCA function on training
    # and testing set of X component
    pca = PCA(n_components=n_sens)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # explained_variance = pca.explained_variance_ratio_
    classifier = LogisticRegression(random_state=0, solver='liblinear',
                                    multi_class='auto')
    classifier.fit(X_train, y_train)

    # Predicting the test set result using
    # predict function under LogisticRegression
    y_pred = classifier.predict(X_test)

    # making confusion matrix between
    # test set of Y and predicted value.
    cm = confusion_matrix(y_test, y_pred)

    return cm


def _multiple_simulations(n_sensor_list, n_bac_list, response_limits,
                          indist, noise):
    """Multiple simulations of a set sensors and bacteria, each time the
    sensors are initiated and sequencially added. The number of sensors equals
    the number of principal components for the PCA. The simulated experiment is
    repeated a minimal amout of time to pefrom the PCA.

    Parameters
    ----------
    n_sensors_list : lst
        A list of different number of sensors which are squencially added to
        the simulated exeriment. It must be a list of integers.
    n_bac_list : lst
    A list of different number of bacteria which are squencially added to
        the simulated exeriment. It must be a list of integers.
    response_limit : tuple
        minimal and maximal sensor response.
    indist : float
        float number in an interval [0.0, 1.0]. Percentage of sensors which
        respond the same to two different bacteria.

    Returns
    -------
    result_dict : dict
        dictionary containing the results of the PCA and the parameters for the
        simulation

        parameters : array
        In parameters one finds under index i: 0: number of sensors, 1: number
        of bactera (equal to the key 'bacterium_number'), 2: percentage of
        indistinguishable sensors, 3: number of measurement repetitions,
        4: percentage of maximum noise, 5: trace of confusion matrix,
        6: trace of confusion matrix / sum of confusion matrix
            """
    results_dict = {}
    for b in n_bac_list:
        # repeats needs to be an integer, and to get the minimum number for
        # PCA it always needs to be rounded up
        repeats = int(math.ceil(2 * n_sensor_list.max() / b))
        # To ensure that there are always test cases of each principal
        # sorting criterium in training set there is a minimum number of
        # repeats
        if repeats < 3:
            repeats = 3

        # Initiate dictionary and arrays for later storage
        results_dict[b] = {}
        parameter_array = np.zeros((7, len(n_sensor_list)))
        variance_matrix = np.zeros((n_sensor_list.max(), len(n_sensor_list)))
        confusion_matrix = np.zeros((b, b, len(n_sensor_list)))
        confusion_matrix = np.zeros((b, b, len(n_sensor_list)))
        x_pca = np.zeros((repeats * b, n_sensor_list.max(),
                         len(n_sensor_list)))
        n_bac = np.zeros((repeats * b, len(n_sensor_list)))

        # Data cube intiated
        data_cube = experiment_repetition(n_sensor_list.max(), b,
                                          response_limits, indist, repeats,
                                          noise)
        data_cube = data_cube.reshape(n_sensor_list.max(), b * repeats).T
        for i, s in enumerate(n_sensor_list, start=0):
            print(f'n_bac: {b}, n_sens: {s}',)
            # Save parameters
            parameter_array[0, i] = s
            parameter_array[1, i] = b
            parameter_array[2, i] = indist
            parameter_array[3, i] = repeats
            parameter_array[4, i] = noise

            # prepare DataFrame
            dataset = pd.DataFrame(data=data_cube[:, :s],
                                   columns=[f'Sensor {e}'for e in
                                            np.arange(1, s+1)])

            # Add number of bacterium to DataFrame
            bac_number = np.zeros(b * repeats)
            for j in np.arange(b):
                bac_number[j * repeats:(j+1) * repeats] = j
            dataset['bacterium number'] = bac_number
            n_bac[:b * repeats, i] = bac_number

            # Calulate variance matrix of PCA
            variance_matrix[:s, i], x_pca[:b * repeats, :s, i] = simple_PCA(
                    dataset, s)

            # Calculate confusion matrix which shows how many test cases were
            # correctly assigned
            cm = perform_LDA(dataset, s)
            K, L = cm.shape
            confusion_matrix[:K, :L, i] = cm
            parameter_array[5, i] = confusion_matrix[:, :, i].trace()
            parameter_array[6, i] = confusion_matrix[
                    :, :, i].trace() / confusion_matrix[:, :, i].sum()
        results_dict[b]['parameters'] = parameter_array
        results_dict[b]['variance_matrix'] = variance_matrix
        results_dict[b]['confusion_matrix'] = confusion_matrix
        results_dict[b]['bacterium_number'] = n_bac
        results_dict[b]['pca_data'] = x_pca
    return results_dict


def multiple_simulations(n_sensor_list, n_bac_list, response_limits,
                         indist, noise, repeats=False):
    """Multiple simulations of a multiple sensors and bacteria, each time the
    sensors are initiated and sequencially added. The number of sensors equals
    the number of principal components for the PCA. The simulated experiment is
    repeated a minimal amout of time to pefrom the PCA if rep is False,
    otherwise a value can be chosen or is derived from the number of sensors
    and bacteria in the input lists.

    Parameters
    ----------
    n_sensors_list : lst
        A list of different number of sensors which are squencially added to
        the simulated exeriment. It must be a list of integers.
    n_bac_list : lst
    A list of different number of bacteria which are squencially added to
        the simulated exeriment. It must be a list of integers.
    response_limit : tuple
        minimal and maximal sensor response.
    indist : float
        float number in an interval [0.0, 1.0]. Percentage of sensors which
        respond the same to two different bacteria.
    repeats : int, bool, optional
        Number of experiment repetitions. Default is False, no number of
        repetitions is set, and the minium number for each sensor/bacteria set
        is used.

    Returns
    -------
    result_dict : dict
        dictionary containing the results of the PCA and the parameters for the
        simulation

        parameters : array
        In parameters one finds under index i: 0: number of sensors, 1: number
        of bactera (equal to first dict key), 2: percentage of indistinguish-
        able sensors, 3: number of measurement repetitions, 4: percentage of
        maximum noise, 5: trace of confusion matrix 6: trace of confusion
        matrix / sum of confusion matrix
    """
    if repeats is False:
        results_dict = _multiple_simulations(n_sensor_list, n_bac_list,
                                             response_limits, indist, noise)
        return results_dict
    results_dict = {}
    # repeats needs to be an integer, and to get the minimum number for
    # PCA it always needs to be rounded up, the same number of repeats is
    # always used
    rep = int(math.ceil(2 * n_sensor_list.max() / n_bac_list.min()))
    if isinstance(rep, int):
        if repeats > rep:
            rep = repeats
        else:
            warn('Number of repetitions to small to create dataset for PCA'
                 'for all sets of sensor and bacterias specified.')

    for b in n_bac_list:
        # Initiate dictionary and arrays for later storage
        results_dict[b] = {}
        parameter_array = np.zeros((7, len(n_sensor_list)))
        variance_matrix = np.zeros((n_sensor_list.max(), len(n_sensor_list)))
        confusion_matrix = np.zeros((b, b, len(n_sensor_list)))
        confusion_matrix = np.zeros((b, b, len(n_sensor_list)))

        x_pca = np.zeros((rep * b, n_sensor_list.max(),
                         len(n_sensor_list)))
        n_bac = np.zeros((rep * b, len(n_sensor_list)))

        # initiate datacube
        data_cube = experiment_repetition(n_sensor_list.max(), b,
                                          response_limits, indist,
                                          rep, noise)
        data_cube = data_cube.reshape(n_sensor_list.max(), b * rep).T
        for i, s in enumerate(n_sensor_list, start=0):
            print(f'n_bac: {b}, n_sens: {s}',)
            # To ensure that there are always test cases of each principal
            # sorting criterium in training set there is a minimum number of
            # repeats

            # Save parameters
            parameter_array[0, i] = s
            parameter_array[1, i] = b
            parameter_array[2, i] = indist
            parameter_array[3, i] = rep
            parameter_array[4, i] = noise

            # prepare DataFrame
            dataset = pd.DataFrame(data=data_cube[:, :s],
                                   columns=[f'Sensor {e}'for e in
                                            np.arange(1, s+1)])

            # Add number of bacterium to DataFrame
            bac_number = np.zeros(b * rep)
            for j in np.arange(b):
                bac_number[j * rep:(j+1) * rep] = j
            dataset['bacterium number'] = bac_number
            n_bac[:b * rep, i] = bac_number

            # Calulate variance matrix of PCA
            variance_matrix[:s, i], x_pca[:b * rep, :s, i] = simple_PCA(
                    dataset, s)

            # Calculate confusion matrix which shows how many test cases were
            # correctly assigned
            cm = perform_LDA(dataset, s)
            K, L = cm.shape
            confusion_matrix[:K, :L, i] = cm
            parameter_array[5, i] = confusion_matrix[:, :, i].trace()
            parameter_array[6, i] = confusion_matrix[
                    :, :, i].trace() / confusion_matrix[:, :, i].sum()
        results_dict[b]['parameters'] = parameter_array
        results_dict[b]['variance_matrix'] = variance_matrix
        results_dict[b]['confusion_matrix'] = confusion_matrix
        results_dict[b]['bacterium_number'] = n_bac
        results_dict[b]['pca_data'] = x_pca
    return results_dict

