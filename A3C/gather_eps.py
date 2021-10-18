class gather_eps:
  def __init__(self):
    super(gather_eps,self).__init__()
    self.dict_list={'number_of_episodes':[],'score':[],'loss':[]}

  def append_data(self,episode,score,loss):
    self.dict_list['number_of_episodes'].append(episode)#,d_score,d_loss.item()
    self.dict_list['score'].append(score)
    self.dict_list['loss'].append(loss)

  def print_dict(self):
    print(self.dict_list)
