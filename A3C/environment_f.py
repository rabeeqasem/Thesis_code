import ctypes
import itertools
import json
import math
import os
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm


class environment:
  def __init__(self,g):
      super(environment, self).__init__()
      self.g=g
      self.enc_node={}
      self.dec_node={}

      for index,nd in enumerate(self.g.nodes):
        self.enc_node[nd]=index
        self.dec_node[index]=nd
            
  def state_enc(self,dst, end):
    n=len(self.g.nodes)
    return dst+n*end

  def state_dec(self,state):
    n=len(self.g.nodes)
    dst = state%n
    end = (state-dst)/n
    return dst, int(end)
  def reset(self):
    self.state=self.state_enc(self.enc_node[1130166767],self.enc_node[1731824802])
    return self.state
  

  def step(self,state,action):
    done=False    
    current_node , end = self.state_dec(state)

    new_state = self.state_enc(action,end)
    rw,link=self.rw_function(current_node,action)

    if not link:
        new_state = state
        return new_state,rw,False  

    elif action == end:
        rw = 10000 #500*12
        done=True
      
    return new_state,rw,done
  
  def wayenc(self,current,new_state,type=1):
    #encoded
    if type==1: #distance
      if new_state in self.g[current]:
        #rw=data[g[current][new_state]['parent']]['distance']*-1
        rw=self.g[current][new_state]['weight']*-1
        return rw,True
      #rw=int(-sys.maxsize - 1)
      rw=-500
      return rw,False

  def rw_function(self,current_node,new_node):
    beta=1 #between 1 and 0
    current_id=self.dec_node[current_node]
    new_id=self.dec_node[new_node]
    rw0,link=self.wayenc(current_id,new_id)

    rw1=0
    
    frw=rw0*beta+(1-beta)*rw1
    return frw,link

  def state_to_vector(self,current_node,end_node):
    n=len(self.g.nodes)
    source_state_zeros=[0.]*n
    source_state_zeros[current_node]=1.

    end_state_zeros=[0.]*n
    end_state_zeros[end_node]=1.
    vector=source_state_zeros+end_state_zeros
    return vector


    return frw,link

print('rabeee')
