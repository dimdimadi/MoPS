#!/usr/bin/env python
# coding: utf-8

# In[310]:


import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (14.0, 8.0)


# In[311]:


#Defining the Parameter

A = 0.3   #Mean Albedo of Earth Surface 
S = 1366  #Solar constant in W/m2
sigma = 5.67 * 1e-8  #Stefan-Boltzmann constant in W/m2K4

#Short wave radiation

t_a = 0.53 #transmission of the atmosphere for short wave radiation
a_s = 0.19 #surface albedo for short wave radiation
a_a = 0.3  #albedo of the atmosphere for short wave radiation

#Long wage radiation

t_A = 0.06 #transmission of the atmosphere for long wave radiation
a_A = 0.31 #albedo of the atmosphere for long wave radiation

c = 2.7  # Wm-2K-1


# In[312]:


#Set of energy balance equations (no atmosphere)
T1 = (S * (1 - A) / (4 * sigma))**(.25)
print('Mean temperature of the earth assuming no atmosphere: %.3f' % T1)


# In[313]:


#Set of energy balance equations (with atmosphere):


def f(t):
    T_a = t[0]
    T_s = t[1]

    F = np.zeros((2,))
    F[0] = -(t_a) * (1 - a_s) * S / 4 + c * (T_s - T_a) + sigma * T_s**4 * (1 - a_A) - sigma * T_a**4
    F[1] = (-(1 - a_a - t_a + a_s * t_a)) * S / 4 - c * (T_s - T_a) - sigma * T_s**4 * (1 - t_A - a_A) + 2 * sigma * T_a**4

    return F

T2 = fsolve(f, [0, 0])[1]
print('Mean temperature of the earth assuming with atmosphere: %.3f' % T2)


# In[306]:


#Simulation of Relationship between Mean Temperature and Solar Constant

S_range = S * np.arange(0.8, 1.2, 0.01)

T2 = []
for S in S_range:
    T2.append(fsolve(model, [0, 0])[1])
    
plt.xlabel('Solar Constant (W/m2)', fontsize=14)
plt.ylabel('Mean Temperature (K)', fontsize=14)
plt.title('Relationship between Mean Temperature and Solar Constant', fontsize=16) 
plt.grid(True)
seaborn.scatterplot(S_range, T2)


# In[307]:


#Implementation of glaciations mechanism in the model ( Surface albedo depend on the temperature).
#surface albedo decreasing and increasing constant as = 0.19 -> 0.65

T_glaciation = []
as_range = np.arange(0.19, 0.65, 0.01)
for a_s in as_range:
    T_glaciation.append(fsolve(model, [0, 0])[1])

plt.xlabel('Albedo Surface', fontsize=14)
plt.ylabel('Temperature (K)', fontsize=14)
plt.title('Implementation of glaciations mechanism in the model ( Surface albedo depend on the temperature).', fontsize=16) 
plt.grid(True)
seaborn.scatterplot(as_range, T_glaciation)
plt.show()


# In[308]:


#Calculation of solar constant values associated with glacial-interglacial transition of the Earth system.

S_range2 = np.arange(1200, 1400, 1)
T2 = []
for S in S_range2:
    T2.append(fsolve(model, [0, 0])[1])

plt.xlabel('Solar Constant (W/m2)', fontsize=14)
plt.ylabel('Mean Temperature during Glaciations (K)', fontsize=14)
plt.title('Solar constant values associated with glacial-interglacial transition of the Earth system', fontsize=16) 
plt.grid(True)
seaborn.scatterplot(S_range2,T2)
plt.show()
