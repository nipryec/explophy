##################################
#
# Little script to simulate delta formation
#
# version : 22/12/2024
#
# Author : nipryec
#

###
# Imports

import numpy as np
import matplotlib.pyplot as plt

###
# User settings

# size of the environnement
dim = 10

# duration of the experiment
duration = 100
# saving points
save_time = [np.exp(i) for i in range(int(np.log(duration)))]

###
# dev settings

# compute or not, not available yet
# compute = True

###
# initialization

ground = np.ones((dim,dim))
mid = dim//2

for i in range(dim):
	ground[i, mid] = 0

###
# main 

print(ground)










