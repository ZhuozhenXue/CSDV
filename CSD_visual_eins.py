# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:53:58 2026

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



#%%
#np.save('D:/PythonCode/CSDV/J_CSD_TS.npy', J)


#%% Steady-state points
import numpy as np
#J = np.load('D:/PythonCode/EWS/J_CSD_TS.npy')
k_m = 2000
def power_method_stable(A, v_ini, max_iter = k_m):
    n = A.shape[0]
    v = np.zeros((n, max_iter))
    v[:,0] = v_ini
    
    for k in range(max_iter-1):
        v[:,k+1] = A @ v[:,k]
    return v.T

m = 30
Vss, Vtp = np.zeros((k_m,2,m)), np.zeros((k_m,2,m))


theta = np.linspace(0, 2*np.pi, m)
r = 0.1
v0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

'''
# Change Vss here: 0, 1, 2
'''
for j in range(m):
    Vss[:,:,j] =  power_method_stable(J[:,:,0],  v0[j,:], max_iter = k_m)#   <-------------- Here!
    Vtp[:,:,j] =  power_method_stable(J[:,:,-1], v0[j,:], max_iter = k_m)

#%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

def plot_with_arrows(points, arrow_step=5, color=None, label=None):
   
    # Create line segments and use color mapping
    n_points = len(points)
    
    # Create list of segments
    segments = np.array([[[points[i, 0], points[i, 1]], 
                          [points[i+1, 0], points[i+1, 1]]] 
                         for i in range(n_points - 1)])
    
    # Create color values (based on segment index)
    t_values = (np.arange(n_points - 1) / (n_points - 2)) ** 0.45
    
    # Create color mapping
    light_blue = np.array([0.902, 0.941, 0.980])
    dark_blue  = np.array([0.039, 0.251, 0.439])
    
    # Interpolate colors
    colors = light_blue + np.outer(t_values, dark_blue - light_blue)
    
    # Create LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=4, zorder=1)
    
    # Add to figure
    plt.gca().add_collection(lc)
    
    # Set axis ranges
    plt.xlim(points[:, 0].min(), points[:, 0].max())
    plt.ylim(points[:, 1].min(), points[:, 1].max())
    
    # Add direction arrows
    #'''
    AC = 4
    arrow_count = 0
    arrow_color = '#197FCD'
    for i in range(0, len(points) - 1, arrow_step):
        if arrow_count >= AC:
            break
        
        dx = points[i+1, 0] - points[i, 0]
        dy = points[i+1, 1] - points[i, 1]
        
        plt.arrow(points[i, 0], points[i, 1], dx, dy,
                 head_width=0.005, head_length=0.01,
                 fc=arrow_color, ec=arrow_color, zorder=2)
        
        arrow_count += 1
    #'''
    
'''
#######################Main program#######################
'''
plt.figure(figsize=(6, 4))
# Plot all trajectories
for j in range(m):
    plot_with_arrows(Vss[:, :, j], arrow_step=50)
    
plt.xlim([-0.12,0.12])
plt.ylim([-0.12,0.12])

# Thicken all borders
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('#999999')
ax.tick_params(axis='both', color='#999999', width=2, length=6)
# Axis ticks
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
# Set tick label font size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

#plt.xlabel(r'A', fontsize=18)
#plt.ylabel(r'B', fontsize=18)


plt.gca().set_rasterized(True)


plt.show()


#%% Critical point
import numpy as np
#J = np.load('D:/PythonCode/CSDV/J_CSD_TS.npy')
k_m = 2000
def power_method_stable(A, v_ini, max_iter = k_m):
    n = A.shape[0]
    v = np.zeros((n, max_iter))
    v[:,0] = v_ini
    
    for k in range(max_iter-1):
        v[:,k+1] = A @ v[:,k]
    return v.T

m = 30
Vss, Vtp = np.zeros((k_m,2,m)), np.zeros((k_m,2,m))


theta = np.linspace(0, 2*np.pi, m)
r = 0.1
v0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

for j in range(m):
    Vss[:,:,j] =  power_method_stable(J[:,:,0],  v0[j,:], max_iter = k_m)
    Vtp[:,:,j] =  power_method_stable(J[:,:,-1], v0[j,:], max_iter = k_m)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

def plot_with_arrows(points, arrow_step=5, color=None, label=None):
   
    # Create line segments and use color mapping
    n_points = len(points)
    
    # Create list of segments
    segments = np.array([[[points[i, 0], points[i, 1]], 
                          [points[i+1, 0], points[i+1, 1]]] 
                         for i in range(n_points - 1)])
    
    # Create color values (based on segment index)
    t_values = (np.arange(n_points - 1) / (n_points - 2)) ** 0.45
    
    # Create color mapping
    light_red = np.array([0.9961, 0.9412, 0.9333])
    dark_red  = np.array([0.6588, 0.1961, 0.1569])
    
    # Interpolate colors
    colors = light_red + np.outer(t_values, dark_red - light_red)
    
    # Create LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=4, zorder=1)
    
    # Add to figure
    plt.gca().add_collection(lc)
    
    # Set axis ranges
    plt.xlim(points[:, 0].min(), points[:, 0].max())
    plt.ylim(points[:, 1].min(), points[:, 1].max())
    
    # Add direction arrows
    #'''
    AC = 4
    arrow_count = 0
    arrow_color = '#DC2626'
    for i in range(0, len(points) - 1, arrow_step):
        if arrow_count >= AC:
            break
        
        dx = points[i+1, 0] - points[i, 0]
        dy = points[i+1, 1] - points[i, 1]
        
        plt.arrow(points[i, 0], points[i, 1], dx, dy,
                 head_width=0.005, head_length=0.01,
                 fc=arrow_color, ec=arrow_color, zorder=2)
        
        arrow_count += 1
    #'''
    
'''
#######################Main program#######################
'''
plt.figure(figsize=(6, 4))
# Plot all trajectories
for j in range(m):
    plot_with_arrows(Vtp[:, :, j], arrow_step=50)
    
plt.xlim([-0.12,0.12])
plt.ylim([-0.12,0.12])

# Thicken all borders
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('#999999')
ax.tick_params(axis='both', color='#999999', width=2, length=6)
# Axis ticks
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
# Set tick label font size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

#plt.xlabel(r'A', fontsize=18)
#plt.ylabel(r'B', fontsize=18)

plt.gca().set_rasterized(True)

plt.show()