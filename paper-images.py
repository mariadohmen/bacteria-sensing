# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:38:39 2020

@author: Maria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sim_functions import (multiple_simulations)
# %%

NOISE = 0.10
INDIST = 0.4
RESPONSE_LIMITS = (0.7, 1.7)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)


sim_list = np.zeros(40)
for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)

# %%

# LDA Fig
fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()

# Variance ratio figure

index = 7

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
plt.tight_layout()
plt.plot()

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp_main-fig-2.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../variance-ratio-LDA_paper_sort_const-r_sen-eq-comp.xlsx')
# %%

NOISE = 0.10
INDIST = 0.4
RESPONSE_LIMITS = (0.5, 2)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)

sim_list = np.zeros(40)

for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)

# %%

# LDA Fig
fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()

index = 7

# Variance ratio figure

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
# plt.savefig(f'../variance-ratio-LDA_paper_sort_const-r_sen-eq-comp.png', dpi=500)

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../variance-ratio-LDA_paper_sort_const-r_sen-eq-comp.xlsx')

# %%

NOISE = 0.1
INDIST = 0.2
RESPONSE_LIMITS = (0.7, 1.7)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)

sim_list = np.zeros(40)

for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)

# %%

fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()


index = 7

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
plt.tight_layout()
plt.plot()

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp_fall-4.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../variance-ratio-PCA_paper_const_min-r_sen-eq-comp_fall-4.xlsx')

# %%

NOISE = 0.1
INDIST = 0.6
RESPONSE_LIMITS = (0.7, 1.7)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)

sim_list = np.zeros(40)

for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)

# %%

fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()


index = 7

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
plt.tight_layout()
plt.plot()

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp_fall-5.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../variance-ratio-PCA_paper_const_min-r_sen-eq-comp_fall-5.xlsx')

# %%

NOISE = 0.05
INDIST = 0.4
RESPONSE_LIMITS = (0.7, 1.7)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)

sim_list = np.zeros(40)

for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)
# %%

fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()


index = 7

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
plt.tight_layout()
plt.plot()

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp_fall-3.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../s variance-ratio-LDA_paper_sort_const-r_sen-eq-comp_fall-3.xlsx')
# %%

NOISE = 0.15
INDIST = 0.4
RESPONSE_LIMITS = (0.7, 1.7)
# Minimum number of repeated measurements dictated by components in PCA,
# doubled for LDA: 2 * n_sens = n_bac * repeats

n_sensor_list = np.arange(2, 26, 1)
n_bac_list = np.arange(2, 11, 2)
sim_list = np.zeros(40)

for i in np.arange(sim_list):
    sim_list[i] = multiple_simulations(n_sensor_list, n_bac_list,
                                       RESPONSE_LIMITS, INDIST, NOISE, True)
# %%

fig1 = plt.figure(figsize=(9, 8))

ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.set_xticklabels([])

plt.sca(ax0)

cm_trace_mean = np.zeros((len(sim_list),
                          len(sim_list[0][2]['parameters'][6, :]),
                          len(n_bac_list)))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        cm_trace_mean[i, :, j] = res[b]['parameters'][6, :]

for i, b in enumerate(n_bac_list):
    plt.plot(sim_list[0][b]['parameters'][0, :],
             (cm_trace_mean.mean(axis=0)[:, i] - 1 / b) / (1 - 1 / b),
             label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('correct classification')
plt.legend()

plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(sim_list[0][b]['parameters'][0, :],
                cm_trace_mean.std(axis=0)[:, i] * b / (b-1),
                label=f'Number of bacteria: {b}')

plt.ylabel('error')
plt.grid()
plt.xlabel('number of sensors')
plt.plot()


index = 7

fig2 = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax0.set_xticklabels([])

variance_mean = np.zeros((len(sim_list), len(n_bac_list),
                          *sim_list[0][2]['variance_matrix'].shape))
for i, res in enumerate(sim_list):
    for j, b in enumerate(n_bac_list):
        variance_mean[i, j, :, :] = res[b]['variance_matrix']

plt.sca(ax0)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.mean(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('variance ratio')
plt.legend()
plt.title('Variance ratio when using {} sensors'.format(int(
        sim_list[0][2]['parameters'][0, index])))
plt.sca(ax1)
for i, b in enumerate(n_bac_list):
    plt.scatter(np.arange(1, sim_list[0][b]['parameters'][0, index] + 1),
                variance_mean.std(axis=0)[i, :int(
                        sim_list[0][b]['parameters'][0, index]), index],
                label=f'Number of bacteria: {b}')
plt.grid()
plt.ylabel('error')
plt.xlabel('principal components')
plt.tight_layout()
plt.plot()

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = sim_list[0][b]['parameters'][0, :]
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = cm_trace_mean.mean(axis=0)[:, i]
    data[f'std (n_b: {b})'] = cm_trace_mean.std(axis=0)[:, i]
# data.to_excel('../quality-LDA_paper_sort_const-r_sen-eq-comp_fall-2.xlsx')

# put it into dataframe
data = pd.DataFrame()
data['n_sensors'] = np.arange(1, sim_list[0][2]['parameters'][0, index] + 1)
for i, b in enumerate(n_bac_list):
    data[f'mean (n_b: {b})'] = variance_mean.mean(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
    data[f'std (n_b: {b})'] = variance_mean.std(axis=0)[i, :int(
            sim_list[0][b]['parameters'][0, index]), index]
# data.to_excel('../variance-ratio-LDA_paper_sort_const-r_sen-eq-comp_fall-2.xlsx')
