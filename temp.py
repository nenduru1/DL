# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Boltzmann MAchine
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

#prepare train and test 

training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_Set=np.array(training_set,dtype='int')
testing_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
training_Set=np.array(training_set,dtype='int')