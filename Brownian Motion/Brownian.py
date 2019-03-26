#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (14.0, 8.0)


# In[10]:


n_step = 1000 # Number of time steps
n_parts = 100 # Number of particle

sigma = 1.  # Standard deviation
mu = 0.  # Mean

x = np.zeros((n_step, n_parts)) # Initial state of x
y = np.zeros((n_step, n_parts)) # Initial state of y


for i in range(n_step - 1): # Itteration
    x[i + 1, :] = x[i, :] + np.random.normal(mu, sigma, (1, n_parts))
    y[i + 1, :] = y[i, :] + np.random.normal(mu, sigma, (1, n_parts))


# In[11]:


plt.plot(x, y) # Plot the 2D Trajectory
plt.plot(x, y, 'oy') # Mark the following position with yellow dot
plt.plot(x[0], y[0], 'og') # Mark the start position with green dot
plt.plot(x[-1], y[-1], 'or') # Mark the final position with red dot
plt.xlabel('x displacement', fontsize=14)
plt.ylabel('y displacement', fontsize=14)
plt.title('Brownian Motion of Multiple particles in 2D in N step time', fontsize=16) 
plt.grid(True)


# In[12]:


d_square = x**2 + y**2 
mean_square = np.mean(d_square, axis=1)
plt.plot(mean_square)

#We define time as number of discrete interval or steps (n_steps), so we could assume that t = 1, so time is range of the n_step or N

plt.plot(range(n_step), 'or')
plt.title('Mean Square of Displacement versus Time n_step=1000 n_part=900', fontsize=16)
plt.ylabel('mean square of displacement', fontsize=14)
plt.xlabel('time', fontsize=14)
plt.grid(True)
red_patch = mpatches.Patch(color='red', label='Time')
blue_patch = mpatches.Patch(color='blue', label='Mean Square of Displacement')
plt.legend(handles=[red_patch,blue_patch])


# In[ ]:


plt.plot(d_square)
plt.plot(range(n_step), 'or')

