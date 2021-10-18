
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class a3c_plot:
  def __init__(self,dict_list):
    self.dict_list=dict_list



  def plot_graph(self):
    fig, ax =plt.subplots(1,2,figsize=(15,5))
    sns.lineplot(data=self.dict_list, x="number_of_episodes", y="score",ax=ax[0])
    sns.lineplot(data=self.dict_list, x="number_of_episodes", y="loss",ax=ax[1])

