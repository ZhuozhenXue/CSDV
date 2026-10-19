# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:45:18 2026

@author: 98024
"""

import numpy as np

# Iteration parameters
T = 30
T_E = 20
K = 3000
delta_t = 0.1
eps = np.finfo(float).eps  # Smallest positive floating-point number

# System parameters
k01, k1, J_1, kd1 = 0.06, 0.6, 0.06, 0.6
k02, k2, J_2, kd2 = 0.0006, 0.03, 1.6, 0.09
k3, J_3, kd3 = 17, 0.08, 1.66
k04, k4, J1_4, J2_4, kd4 = 0.0012, 0.012, 0.15, 0.36, 0.035
k05, k5, J_5, kd5 = 0.003, 0.06, 3.5, 0.09
k6, J_6, kd6 = 17, 0.06, 1.66
k07, k7, J1_7, J2_7, kd7 = 0.0002, 0.012, 5, 0.2, 0.035
k81, J1_8, k82, J2_8, kd8 = 1, 0.2, 0.6, 0.5, 0.5
k91, J1_9, k92, J2_9, kd9 = 1, 0.2, 0.6, 0.5, 0.5
n1, n3, n4, n6, n7 = 2, 2, 2, 2, 2

# Bifurcation parameter
TGF0_b = 1.036

# E-state parameters
TGF0_E = np.linspace(TGF0_b-1, TGF0_b, T_E+1)
TGF0_E = np.delete(TGF0_E, 0) 
# E-state system initial values
x_ini_E = np.zeros(9)
x_ini_E[0], x_ini_E[1], x_ini_E[2] = 0.1289*1.1, 0.0125*1.1, 0.0055*1.1
x_ini_E[3], x_ini_E[4], x_ini_E[5] = 0.3764*1.1, 0.0333*1.1, 0.0099*1.1
x_ini_E[6], x_ini_E[7], x_ini_E[8] = 0.3477*1.1, 3.1980*1.1, 0.0020*1.1

# H-state parameters
TGF0_H = np.linspace(TGF0_b, TGF0_b+0.5, T_E-10+1)
TGF0_H = np.delete(TGF0_H, 0)
# H-state system initial values
x_ini_H = np.zeros(9)
x_ini_H[0], x_ini_H[1], x_ini_H[2] = 0.1323*1.1, 0.1285*1.1, 1.0366*1.1
x_ini_H[3], x_ini_H[4], x_ini_H[5] = 0.0415*1.1, 0.0870*1.1, 0.0287*1.1
x_ini_H[6], x_ini_H[7], x_ini_H[8] = 0.3285*1.1, 1.2679*1.1, 1.9321*1.1

TGF0 = np.concatenate([TGF0_E, TGF0_H])

#%

# Initialize storage arrays
x1, x2, x3 = np.zeros((T+1, K+1)), np.zeros((T+1, K+1)), np.zeros((T+1, K+1))
x4, x5, x6 = np.zeros((T+1, K+1)), np.zeros((T+1, K+1)), np.zeros((T+1, K+1))
x7, x8, x9 = np.zeros((T+1, K+1)), np.zeros((T+1, K+1)), np.zeros((T+1, K+1))
ssx1, ssx2, ssx3 = np.zeros(T), np.zeros(T), np.zeros(T)
ssx4, ssx5, ssx6 = np.zeros(T), np.zeros(T), np.zeros(T)
ssx7, ssx8, ssx9 = np.zeros(T), np.zeros(T), np.zeros(T)

'''
#######################Deterministic model#######################
'''

for i in range(T):  # i represents the sequential pseudo-time node
    if i < 20:
        x_ini = x_ini_E
        # E-state: use x_ini_E for the first point, use previous step's result afterwards
        if i == 0:
            # For the first point, directly use x_ini_E
            x1[i,0], x2[i,0], x3[i,0] = x_ini[0], x_ini[1], x_ini[2]
            x4[i,0], x5[i,0], x6[i,0] = x_ini[3], x_ini[4], x_ini[5]
            x7[i,0], x8[i,0], x9[i,0] = x_ini[6], x_ini[7], x_ini[8]
        else:
            # Use the previous step's computation result
            x1[i,0], x2[i,0], x3[i,0] = 1.1*x1[i-1,-1], 1.1*x2[i-1,-1], 1.1*x3[i-1,-1]
            x4[i,0], x5[i,0], x6[i,0] = 1.1*x4[i-1,-1], 1.1*x5[i-1,-1], 1.1*x6[i-1,-1]
            x7[i,0], x8[i,0], x9[i,0] = 1.1*x7[i-1,-1], 1.1*x8[i-1,-1], 1.1*x9[i-1,-1]
    else:
        x_ini = x_ini_H
        # H-state: reset to x_ini_H at the first point (i=20)
        if i == 20:
            x1[i,0], x2[i,0], x3[i,0] = x_ini[0], x_ini[1], x_ini[2]
            x4[i,0], x5[i,0], x6[i,0] = x_ini[3], x_ini[4], x_ini[5]
            x7[i,0], x8[i,0], x9[i,0] = x_ini[6], x_ini[7], x_ini[8]
        else:
            # Use the previous step's computation result
            x1[i,0], x2[i,0], x3[i,0] = 1.1*x1[i-1,-1], 1.1*x2[i-1,-1], 1.1*x3[i-1,-1]
            x4[i,0], x5[i,0], x6[i,0] = 1.1*x4[i-1,-1], 1.1*x5[i-1,-1], 1.1*x6[i-1,-1]
            x7[i,0], x8[i,0], x9[i,0] = 1.1*x7[i-1,-1], 1.1*x8[i-1,-1], 1.1*x9[i-1,-1]

    for k in range(K):  # Euler-Maruyama iteration
        # Note: do not reassign initial values here again
        
        x1[i,k+1] = x1[i,k] + (k01 + k1/(1+(x7[i,k]/J_1)**n7) 
                               - kd1*x1[i,k]) * delta_t 
        x2[i,k+1] = x2[i,k] + (k02 + k2*(((x1[i,k]+TGF0[i])/J_2)**n1)/(1+((x1[i,k]+TGF0[i])/J_2)**n1) 
                               - kd2*x2[i,k]) * delta_t 
        x3[i,k+1] = x3[i,k] + (k3*x2[i,k]/(1+(x4[i,k]/J_3)**n4) 
                               - kd3*x3[i,k]) * delta_t 
        x4[i,k+1] = x4[i,k] + (k04 + k4/(1+(x3[i,k]/J1_4)**n3+(x6[i,k]/J2_4)**n6) 
                               - kd4*x4[i,k]) * delta_t 
        x5[i,k+1] = x5[i,k] + (k05 + k5*((x3[i,k]/J_5)**n3)/(1+(x3[i,k]/J_5)**n3) 
                               - kd5*x5[i,k]) * delta_t 
        x6[i,k+1] = x6[i,k] + (k6*x5[i,k]/(1+(x7[i,k]/J_6)**n7) 
                               - kd6*x6[i,k]) * delta_t 
        x7[i,k+1] = x7[i,k] + (k07 + k7/(1+(x3[i,k]/J1_7)**n3+(x6[i,k]/J2_7)**n6) 
                               - kd7*x7[i,k]) * delta_t 
        x8[i,k+1] = x8[i,k] + (k81/(1+(x3[i,k]/J1_8)**n3) + k82/(1+(x6[i,k]/J2_8)**n6) 
                               - kd8*x8[i,k]) * delta_t 
        x9[i,k+1] = x9[i,k] + (k91*((x3[i,k]/J1_9)**n3)/(1+(x3[i,k]/J1_9)**n3) + k92*((x6[i,k]/J2_9)**n6)/(1+(x6[i,k]/J2_9)**n6) 
                               - kd9*x9[i,k]) * delta_t 
        
        # If a component becomes negative, set it back to eps
        x = np.array([x1[i,k+1], x2[i,k+1], x3[i,k+1],
                      x4[i,k+1], x5[i,k+1], x6[i,k+1],
                      x7[i,k+1], x8[i,k+1], x9[i,k+1]])
        x[x < 0] = eps
        (x1[i,k+1], x2[i,k+1], x3[i,k+1],
        x4[i,k+1], x5[i,k+1], x6[i,k+1],
        x7[i,k+1], x8[i,k+1], x9[i,k+1]) = x
        
        
    # Record steady state
    ssx1[i], ssx2[i], ssx3[i] = x1[i,-1], x2[i,-1], x3[i,-1]
    ssx4[i], ssx5[i], ssx6[i] = x4[i,-1], x5[i,-1], x6[i,-1]
    ssx7[i], ssx8[i], ssx9[i] = x7[i,-1], x8[i,-1], x9[i,-1]



#%
#% Add noise to data
'''
#######################Add noise to data#######################
'''

# Original time series
TimeSeries_x1, TimeSeries_x2, TimeSeries_x3 = x1[:-1, :], x2[:-1, :], x3[:-1, :]
TimeSeries_x4, TimeSeries_x5, TimeSeries_x6 = x4[:-1, :], x5[:-1, :], x6[:-1, :]
TimeSeries_x7, TimeSeries_x8, TimeSeries_x9 = x7[:-1, :], x8[:-1, :], x9[:-1, :]


Range_x1 = np.ptp(TimeSeries_x1, axis=1)
Range_x2 = np.ptp(TimeSeries_x2, axis=1)
Range_x3 = np.ptp(TimeSeries_x3, axis=1)
Range_x4 = np.ptp(TimeSeries_x4, axis=1)
Range_x5 = np.ptp(TimeSeries_x5, axis=1)
Range_x6 = np.ptp(TimeSeries_x6, axis=1)
Range_x7 = np.ptp(TimeSeries_x7, axis=1)
Range_x8 = np.ptp(TimeSeries_x8, axis=1)
Range_x9 = np.ptp(TimeSeries_x9, axis=1)

# Estimate noise scale
NoiseScale_x1 = np.min(Range_x1)
NoiseScale_x2 = np.min(Range_x2)
NoiseScale_x3 = np.min(Range_x3)
NoiseScale_x4 = np.min(Range_x4)
NoiseScale_x5 = np.min(Range_x5)
NoiseScale_x6 = np.min(Range_x6)
NoiseScale_x7 = np.min(Range_x7)
NoiseScale_x8 = np.min(Range_x8)
NoiseScale_x9 = np.min(Range_x9)

# Add noise
np.random.seed(42)
sigma = 0.1
TimeSeriesNoisy_x1 = TimeSeries_x1 + sigma * NoiseScale_x1 * np.random.randn(*TimeSeries_x1.shape)
TimeSeriesNoisy_x2 = TimeSeries_x2 + sigma * NoiseScale_x2 * np.random.randn(*TimeSeries_x2.shape)
TimeSeriesNoisy_x3 = TimeSeries_x3 + sigma * NoiseScale_x3 * np.random.randn(*TimeSeries_x3.shape)
TimeSeriesNoisy_x4 = TimeSeries_x4 + sigma * NoiseScale_x4 * np.random.randn(*TimeSeries_x4.shape)
TimeSeriesNoisy_x5 = TimeSeries_x5 + sigma * NoiseScale_x5 * np.random.randn(*TimeSeries_x5.shape)
TimeSeriesNoisy_x6 = TimeSeries_x6 + sigma * NoiseScale_x6 * np.random.randn(*TimeSeries_x6.shape)
TimeSeriesNoisy_x7 = TimeSeries_x7 + sigma * NoiseScale_x7 * np.random.randn(*TimeSeries_x7.shape)
TimeSeriesNoisy_x8 = TimeSeries_x8 + sigma * NoiseScale_x8 * np.random.randn(*TimeSeries_x8.shape)
TimeSeriesNoisy_x9 = TimeSeries_x9 + sigma * NoiseScale_x9 * np.random.randn(*TimeSeries_x9.shape)



#%% Smoothing
'''
#######################Smoothing#######################
'''
import torch
import torch.nn.functional as F

def smooth_matrix_rows(x, k=7):
    # Check input type; if numpy, convert to tensor
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x).float()  # Convert to torch float32
    """
    Apply 1-D moving average smoothing to each row (each time series) of the matrix.
    
    Parameters:
        x : (N, T)  - N time series, each of length T
        k : odd window size (default 7)
    
    Returns:
        (N, T) - same shape, each row smoothed
    """
    N, T = x.shape
    pad = k // 2
    weight = torch.ones(1, 1, k, dtype=x.dtype, device=x.device) / k
    out = torch.empty_like(x)

    for i in range(N):
        xi = x[i].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        xi = F.pad(xi, (pad, pad), mode='replicate')  # Edge padding
        xi = F.conv1d(xi, weight)  # (1, 1, T)
        out[i] = xi.squeeze()
        
    # If input is numpy, convert output back to numpy
    if is_numpy:
        out = out.numpy()
        
    return out

TimeSeriesSmoothed_x1 = smooth_matrix_rows(TimeSeriesNoisy_x1) 
TimeSeriesSmoothed_x2 = smooth_matrix_rows(TimeSeriesNoisy_x2) 
TimeSeriesSmoothed_x3 = smooth_matrix_rows(TimeSeriesNoisy_x3) 
TimeSeriesSmoothed_x4 = smooth_matrix_rows(TimeSeriesNoisy_x4) 
TimeSeriesSmoothed_x5 = smooth_matrix_rows(TimeSeriesNoisy_x5) 
TimeSeriesSmoothed_x6 = smooth_matrix_rows(TimeSeriesNoisy_x6) 
TimeSeriesSmoothed_x7 = smooth_matrix_rows(TimeSeriesNoisy_x7) 
TimeSeriesSmoothed_x8 = smooth_matrix_rows(TimeSeriesNoisy_x8) 
TimeSeriesSmoothed_x9 = smooth_matrix_rows(TimeSeriesNoisy_x9) 



#%% Estimate Jacobian matrix
# Distance between transient and equilibrium point
Dist_x1 = TimeSeriesSmoothed_x1[:,0:K] - TimeSeriesSmoothed_x1[:,-1].reshape(-1, 1)
Dist_x2 = TimeSeriesSmoothed_x2[:,0:K] - TimeSeriesSmoothed_x2[:,-1].reshape(-1, 1)
Dist_x3 = TimeSeriesSmoothed_x3[:,0:K] - TimeSeriesSmoothed_x3[:,-1].reshape(-1, 1)
Dist_x4 = TimeSeriesSmoothed_x4[:,0:K] - TimeSeriesSmoothed_x4[:,-1].reshape(-1, 1)
Dist_x5 = TimeSeriesSmoothed_x5[:,0:K] - TimeSeriesSmoothed_x5[:,-1].reshape(-1, 1)
Dist_x6 = TimeSeriesSmoothed_x6[:,0:K] - TimeSeriesSmoothed_x6[:,-1].reshape(-1, 1)
Dist_x7 = TimeSeriesSmoothed_x7[:,0:K] - TimeSeriesSmoothed_x7[:,-1].reshape(-1, 1)
Dist_x8 = TimeSeriesSmoothed_x8[:,0:K] - TimeSeriesSmoothed_x8[:,-1].reshape(-1, 1)
Dist_x9 = TimeSeriesSmoothed_x9[:,0:K] - TimeSeriesSmoothed_x9[:,-1].reshape(-1, 1)

# 3D array
Dist_3d = np.stack([Dist_x1, Dist_x2, Dist_x3,
                    Dist_x4, Dist_x5, Dist_x6,
                    Dist_x7, Dist_x8, Dist_x9], axis=0) # (9, T, K)
Dist_3d = Dist_3d.transpose(0, 2, 1)                    # (9, K, T)

Y_3d = Dist_3d[:,0:K-1,:]
Z_3d = Dist_3d[:,1:K,:]

J = np.zeros((9,9,T))
Jc_infer = np.zeros((9,9,T))
EV = np.zeros((T,9))


#'''
lamda = 1e-5  # Regularization parameter
for i in range(T):
    J[:,:,i] = Z_3d[:,:,i] @ Y_3d[:,:,i].T @ np.linalg.inv(Y_3d[:,:,i] @ Y_3d[:,:,i].T + lamda * np.eye(9))
    EV[i,:] = np.linalg.eigvals(J[:,:,i])
    #Jc_infer[:,:,i] = (J[:,:,i] - np.eye(9))/delta_t
#'''


#% Power method for dominant eigenvalue
def power_method_stable(A, max_iter=5000):
    n = A.shape[0]
    v = np.zeros((n, max_iter))
    v[:,0] = np.ones(n)
    
    for k in range(max_iter-1):
        v[:,k+1] = A @ v[:,k]
    return v[:,max_iter-1]

V = np.zeros((T,9))
C = np.zeros((9,9,T))
Cd = np.zeros((9,9,T))
for i in range(T):
    V[i,:] = power_method_stable(J[:,:,i])
    #C[:,:,i] = np.linalg.matrix_power(J[:,:,i], 3000)
    #Cd[:,:,i] = np.linalg.matrix_power(Jd[:,:,i], 3000)
    
V_overflow = V[-1,:]
V_norm_1 = np.linalg.norm(V, ord=1, axis=1)


#%%
#np.save('D:/PythonCode/CSDV/J_EMT_EH_010.npy', J)

#%%
import numpy as np
#J = np.load('D:/PythonCode/EWS/J_EMT_EH2_010.npy')

T = 30

# Bifurcation parameter
TGF0_b = 1.036
TGF0_0 = TGF0_b - 1
TGF0 = np.linspace(TGF0_0, TGF0_b+0.5, T+1)
TGF0 = np.delete(TGF0, 0) # T_bifur+1 points, delete the first one
TGF0 = TGF0[10:25]

#%
def power_method_stable(A, max_iter):
    n = A.shape[0]
    v = np.zeros((n, max_iter))
    v[:,0] = np.ones(n)
    
    for k in range(max_iter-1):
        v[:,k+1] = A @ v[:,k]
    return v[:,max_iter-1]

V1 = np.zeros((T,9))
V2 = np.zeros((T,9))
V3 = np.zeros((T,9))

for i in range(T):
    V1[i,:] = power_method_stable(J[:,:,i], max_iter=5000)
    V2[i,:] = power_method_stable(J[:,:,i], max_iter=6000)
    V3[i,:] = power_method_stable(J[:,:,i], max_iter=7000)
    
V1_overflow = V1[-1,:]
V2_overflow = V2[-1,:]
V3_overflow = V3[-1,:]
V1_norm_1 = np.linalg.norm(V1, ord=1, axis=1)
V2_norm_1 = np.linalg.norm(V2, ord=1, axis=1)
V3_norm_1 = np.linalg.norm(V3, ord=1, axis=1)

V1_norm_1 = V1_norm_1[10:25]
V2_norm_1 = V2_norm_1[10:25]
V3_norm_1 = V3_norm_1[10:25]

#% 1-norm of overflow vector
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
# Plot

color1 = '#FFD966'
color2 = '#E8A317'
color3 = '#C68E17'
color4 = '#DC2626'


plt.figure(figsize=(6, 4))

plt.plot(TGF0, V1_norm_1, linewidth=4, label='$k = 5000$', color=color1, marker='o', markersize=8)
plt.plot(TGF0, V2_norm_1, linewidth=4, label='$k = 6000$', color=color2, marker='o', markersize=8)
plt.plot(TGF0, V3_norm_1, linewidth=4, label='$k = 7000$', color=color3, marker='o', markersize=8)
plt.scatter(TGF0[9], V1_norm_1[9],  s=200, marker='*', color=color4, label='Bifurcation point', zorder=5)
plt.scatter(TGF0[9], V2_norm_1[9],  s=200, marker='*', color=color4, zorder=5)
plt.scatter(TGF0[9], V3_norm_1[9],  s=200, marker='*', color=color4, zorder=5)

plt.ylim(-0.2e-2, 4.7e-2)

# Create legend
legend = plt.legend(prop={'size': 14}, 
                    edgecolor='#999999', 
                    frameon=True)
# Thicken border
legend.get_frame().set_linewidth(1)


# Thicken all borders
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('#999999')

# Thicken tick marks
ax.tick_params(axis='both', color='#999999', width=2, length=6)
# Thicken tick labels
for label in ax.get_xticklabels():
    #label.set_fontweight('bold')
    label.set_fontsize(18)
for label in ax.get_yticklabels():
    #label.set_fontweight('bold')
    label.set_fontsize(18)
ax.yaxis.set_major_locator(MultipleLocator(1.5e-2))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# Add title and axis labels
plt.xlabel('TGF0', fontsize=20)
plt.ylabel('CSDV', fontsize=20)
#plt.title('Epithelial to hybrid', fontsize=18, fontweight='bold')


#plt.tight_layout()
plt.subplots_adjust(left=0.15, bottom=0.20, right=0.95, top=0.93)

plt.show()

#%% t-test
from scipy import stats

# Left-tailed test, use alternative='less'
t_stat_V1, pEH_V1 = stats.ttest_1samp(V1_norm_1[0:9], V1_norm_1[9], alternative='less')
t_stat_V2, pEH_V2 = stats.ttest_1samp(V2_norm_1[0:9], V2_norm_1[9], alternative='less')
t_stat_V3, pEH_V3 = stats.ttest_1samp(V3_norm_1[0:9], V3_norm_1[9], alternative='less')
# Output results

print(f'pEH_V1: {pEH_V1:e}')
print(f'pEH_V2: {pEH_V2:e}')
print(f'pEH_V3: {pEH_V3:e}')