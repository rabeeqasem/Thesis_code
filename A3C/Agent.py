
import torch.multiprocessing as mp
from ActorCritics_f import ActorCritics
from a3c_plot_f import a3c_plot

class Agent(mp.Process):
 def __init__(self,global_actor_critic,optimizer,input,n_actions
              ,gamma,lr,worker_name,global_episode_index,env,gather,games,T_max):
   super(Agent,self).__init__()
   self.local_actor_critic=ActorCritics(input,n_actions,env,gamma)
   self.global_actor_critic=global_actor_critic
   self.worker_name='w%02i'%worker_name
   self.episode_idx = global_episode_index
   self.env=env
   self.gather_eps=gather
   self.optimizer=optimizer
   self.N_games=games
   self.T_max=T_max
   self.dict_list={'number_of_episodes':[],'score':[],'loss':[]}

 def list_remember(self,d_episode,d_score,d_loss):
  self.dict_list['number_of_episodes'].append(d_episode)#,d_score,d_loss.item()
  self.dict_list['score'].append(d_score)
  self.dict_list['loss'].append(d_loss.item())
 
 def run(self):
    t_step=1
    max_itr=100
    #self.episode_idx is a gloabl parametar from MP class and we need to get the value from it
    while self.episode_idx.value < self.N_games:



      itr=0
      done=False
      observation=self.env.reset()
      score=0
      penalties=0

      self.local_actor_critic.clear_memory()
      while not done:
        soruce,end=self.env.state_dec(observation)
        action=self.local_actor_critic.choose_action(soruce,end)
        observation_,reward,done=self.env.step(observation,action)
        if reward == -500:
          penalties+=1
        
        score += reward
        self.local_actor_critic.remember(observation,action,reward)

        if t_step% self.T_max==0 or done:
          loss=self.local_actor_critic.calc_loss(done)
          self.optimizer.zero_grad()
          loss.backward()
          #set the current parameters for the workers into the gloabl parameters
          for local_param,global_param in zip(self.local_actor_critic.parameters(),
                                               self.global_actor_critic.parameters()):
            global_param._grad=local_param.grad
          self.optimizer.step()
          self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
          self.local_actor_critic.clear_memory()

        t_step+=1

        itr+=1
        observation=observation_
      print(self.worker_name,'episode',self.episode_idx.value,'reward',score,'penalties',penalties,'goal',done,
        'itr_to_done',itr,'loss',loss.item(),'\n',flush=True)

      self.list_remember(self.episode_idx.value,score,loss)

      self.gather_eps.append_data(self.episode_idx.value,score,loss.item())


      with self.episode_idx.get_lock():
        self.episode_idx.value+=1


    if len(self.dict_list['number_of_episodes'])>0:
      
      plot_g=a3c_plot(self.dict_list)
      plot_g.plot_graph()
