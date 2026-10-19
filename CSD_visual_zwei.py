# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:49:47 2026

@author: 98024
"""
import numpy as np
# Iteration parameters
T = 4
K = 1000
delta_t = 0.01
eps = np.finfo(float).eps  # Smallest positive floating-point number

# System parameters
alpha2   = 3.0

# System initial values
x1_ini, x2_ini = 0.05, 3.0

# Bifurcation parameter range
alpha_b = 9.98
alpha_0 = alpha_b - 9

alpha = np.linspace(alpha_0, alpha_b, T+1)
alpha = np.delete(alpha, 0) # T_bifur+1 points, delete the first one

'''
#######################Deterministic model#######################
'''

# Initialize storage arrays
x1, x2 = np.zeros((T+1, K+1)), np.zeros((T+1, K+1))
ssx1, ssx2 = np.zeros(T), np.zeros(T)
for i in range(T):  # i represents the sequential pseudo-time node
        
    for k in range(K):  # Euler-Maruyama iteration
        ita1, ita2 = np.random.randn(), np.random.randn()
        x1[0,0], x2[0,0] = x1_ini, x2_ini
        
        
        x1[i,k+1] = x1[i,k] + (alpha[i]/(1 + x2[i,k]**3) - x1[i,k]) * delta_t 
        x2[i,k+1] = x2[i,k] + (alpha2 / (1 + x1[i,k]**3) - x2[i,k]) * delta_t 
        
        # If a component becomes negative, set it back to eps
        x = np.array([x1[i,k+1], x2[i,k+1]])
        x[x < 0] = eps
        x1[i,k+1], x2[i,k+1] = x
        
        # Initial value recovery
        x1[i+1,0], x2[i+1,0] = x1[i,-1], x2[i,-1]
    
    # Record steady state
    ssx1[i], ssx2[i] = x1[i,-1], x2[i,-1]



#% Jacobian matrix verification
import autograd.numpy as np
from autograd import jacobian

# Compute Jacobian functions
Jc, J = np.zeros((2,2,T)), np.zeros((2,2,T))
for i in range(T):
    # Define the function
    def f(x):
        return np.array([alpha[i]/(1 + x[1]**3) - x[0], 
                         alpha2 / (1 + x[0]**3) - x[1]])
    jac_f = jacobian(f)
    Jc[:,:,i] = jac_f(np.array([ssx1[i], ssx2[i]]))
    J[:,:,i] = np.eye(2) + delta_t*Jc[:,:,i]

'''
#######################Landscape cross-section#######################
'''

# Steady-state points - merged plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter

#J = np.load('D:/PythonCode/EWS/J_CSD_TS.npy')
k_m = 4000

def power_method_stable(A, v_ini, max_iter = k_m):
    n = A.shape[0]
    v = np.zeros((n, max_iter))
    v[:,0] = v_ini
    
    for k in range(max_iter-1):
        v[:,k+1] = A @ v[:,k]
    return v.T

def plot_ss(V_data, color=None, label=None, alpha=1.0):
    """
    Plot the transformed trajectory
    """
    n_points = len(V_data)
    A = V_data[:, 0]
    k = np.arange(n_points)
    y_transformed = 1.0 / ((k)**2 + 1)
    points_transformed = np.column_stack([A, y_transformed])
    
    segments = np.array([[[points_transformed[i, 0], points_transformed[i, 1]], 
                          [points_transformed[i+1, 0], points_transformed[i+1, 1]]] 
                         for i in range(n_points - 1)])
    
    if color is None:
        t_values = (np.arange(n_points - 1) / (n_points - 2)) ** 0.45
        light_blue = np.array([0.902, 0.941, 0.980])
        dark_blue  = np.array([0.039, 0.251, 0.439])
        colors = light_blue + np.outer(t_values, dark_blue - light_blue)
    else:
        colors = color
    
    lc = LineCollection(segments, colors=colors, linewidth=4, zorder=1, alpha=alpha)
    plt.gca().add_collection(lc)
    
    # Add legend support (create an invisible line)
    if label:
        plt.plot([], [], color=colors[k_m//20] if isinstance(colors, np.ndarray) else colors, 
                linewidth=4, label=label)
    
    return points_transformed

def plot_tp(V_data, color=None, label=None, alpha=1.0):
    """
    Plot the transformed trajectory
    """
    n_points = len(V_data)
    A = V_data[:, 0]
    k = np.arange(n_points)
    y_transformed = 1.0 / ((k)**2 + 1)
    points_transformed = np.column_stack([A, y_transformed])
    
    segments = np.array([[[points_transformed[i, 0], points_transformed[i, 1]], 
                          [points_transformed[i+1, 0], points_transformed[i+1, 1]]] 
                         for i in range(n_points - 1)])
    
    if color is None:
        t_values = (np.arange(n_points - 1) / (n_points - 2)) ** 0.25
        light_red = np.array([0.9961, 0.9412, 0.9333])
        dark_red  = np.array([0.6588, 0.1961, 0.1569])
        colors = light_red + np.outer(t_values, dark_red - light_red)
    else:
        colors = color
    
    lc = LineCollection(segments, colors=colors, linewidth=4, zorder=1, alpha=alpha)
    plt.gca().add_collection(lc)
    
    # Add legend support
    if label:
        plt.plot([], [], color=colors[k_m//20] if isinstance(colors, np.ndarray) else colors, 
                linewidth=4, label=label)
    
    return points_transformed



# Set parameters
m = 2
r = 0.1
theta = np.array([0, np.pi])
v0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

# Create figure
plt.figure(figsize=(5.6, 4))

# Plot first group of data (J[:,:,0] and J[:,:,-1])
for j in range(m):
    Vss1 = power_method_stable(J[:,:,0], v0[j,:], max_iter = k_m)
    if j == 0:  # Add label only once
        plot_ss(Vss1, label='Sampled state')
    else:
        plot_ss(Vss1)


# Plot second group of data (J[:,:,1] and J[:,:,-1])
for j in range(m):
    Vss2 = power_method_stable(J[:,:,1], v0[j,:], max_iter = k_m)
    plot_ss(Vss2)
    
    
for j in range(m):
    Vss3 = power_method_stable(J[:,:,2], v0[j,:], max_iter = k_m)
    plot_ss(Vss3)


for j in range(m):
    Vtp = power_method_stable(J[:,:,-1], v0[j,:], max_iter = k_m)
    if j == 0:  # Add label only once
        plot_tp(Vtp, label='Bifurcation')
    else:
        plot_tp(Vtp)

# Set x-axis range
plt.xlim([-0.037, 0.037])
plt.ylim([(0-0.01)*1e-4, (1+0.01)*1e-4])

# Thicken all borders
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('#999999')
ax.tick_params(axis='both', color='#999999', width=2, length=6)

# Axis ticks
ax.xaxis.set_major_locator(MultipleLocator(0.03))

#'''
# Actual ticks
ax.yaxis.set_major_locator(MultipleLocator(0.5e-4))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#'''




# Set tick label font size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Create legend
legend = plt.legend(prop={'size': 14}, 
                    edgecolor='#999999', 
                    frameon=True, loc='upper center')
# Thicken border
legend.get_frame().set_linewidth(1)


#plt.xlabel(r'A', fontsize=18)
plt.ylabel(r'$1/(k^2+1)$', fontsize=22)

#plt.gca().set_rasterized(True)
plt.tight_layout()


plt.show()