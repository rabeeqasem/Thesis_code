
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch as T

class ActorCritics(nn.Module):
  def __init__(self,input,n_actions,env,gamma=0.99):
    super(ActorCritics,self).__init__()
    self.gamma=gamma
    self.env=env
    self.n_actions=n_actions

    self.pi1=nn.Linear(input,128)
    self.v1=nn.Linear(input,128)

    self.pi2=nn.Linear(128,64)
    self.v2=nn.Linear(128,64)

    self.pi3=nn.Linear(64,32)
    self.v3=nn.Linear(64,32)

    self.pi4=nn.Linear(32,16)
    self.v4=nn.Linear(32,16)

    self.pi5=nn.Linear(16,8)
    self.v5=nn.Linear(16,8)

    self.pi6=nn.Linear(8,4)
    self.v6=nn.Linear(8,4)

    self.pi7=nn.Linear(4,2)
    self.v7=nn.Linear(4,2)

    self.pi=nn.Linear(2,n_actions)
    self.v=nn.Linear(2,1)

    self.rewards=[]
    self.actions=[]
    self.states=[]
  
  #this function takes the values of the state,actions,and reward and append to the memory
  def remember(self,state,action,reward):
    self.actions.append(action)
    self.rewards.append(reward)
    self.states.append(state)
  
  #this function reset the memory each time we are calling the learning function
  def clear_memory(self):
    self.states=[]
    self.actions=[]
    self.rewards=[]

  def forward(self,state):
    pi1=F.relu(self.pi1(state))
    v1=F.relu(self.v1(state))

    pi2=F.relu(self.pi2(pi1))
    v2=F.relu(self.v2(v1))

    pi3=F.relu(self.pi3(pi2))
    v3=F.relu(self.v3(v2))

    pi4=F.relu(self.pi4(pi3))
    v4=F.relu(self.v4(v3))

    pi5=F.relu(self.pi5(pi4))
    v5=F.relu(self.v5(v4))

    pi6=F.relu(self.pi6(pi5))
    v6=F.relu(self.v6(v5))

    pi7=F.relu(self.pi7(pi6))
    v7=F.relu(self.v7(v6))

    pi=self.pi(pi7)
    v=self.v(v7)
    return pi,v
  
  def calc_returns(self,done,vstates):
    # 1-convert the states into tensor
    # 2- send the state into the forward function and get the (policy , value) we are intreseted in the value
    # 3- the return = (the last value in the list wich the value of the terminal state)*(1-done)
    # define a batch return list
    #loop through the inverted reward list
    #return(r)=reward+(gamma*R)
    #append R to the batch return list
    #reverce the batch return list to became the same ordder as the reward list
    '''
    list_state=[]
    if len(self.states)>1:
      for lstate in self.states:
        soruce,end=self.env.state_dec(lstate)
        state_v=self.env.state_to_vector(soruce,end)
        list_state.append(state_v)
      states=T.tensor(list_state)

    else:
      soruce,end=self.env.state_dec(self.states[0])
      state_v=self.env.state_to_vector(soruce,end)
      states=T.tensor(state_v)
    '''
    #
    p,v=self.forward(vstates)

    R=v[-1]*(1-int(done))
    batch_return=[]

    for reward in self.rewards[::-1]:
      R=reward+self.gamma*R
      batch_return.append(R)
    batch_return.reverse()
    batch_return=T.tensor(batch_return,dtype=float)
    return batch_return


  def calc_loss(self,done):
    #states=T.tensor(self.states,dtype=T.float)
    list_state=[]
    if len(self.states)>1:
      for lstate in self.states:
        soruce,end=self.env.state_dec(lstate)
        state_v=self.env.state_to_vector(soruce,end)
        list_state.append(state_v)
      states=T.tensor(list_state)

    else:
      soruce,end=self.env.state_dec(self.states[0])
      state_v=self.env.state_to_vector(soruce,end)
      list_state.append(state_v)
      states=T.tensor([list_state])

    actions=T.tensor(self.actions,dtype=T.float)


    returns=self.calc_returns(done,states)


    p,values=self.forward(states)
    values=values.squeeze()

    critic_loss=(returns-values)**2

    probs=T.softmax(p,dim=1)
    dist=Categorical(probs)
    log_probs=dist.log_prob(actions)
    actor_loss=-log_probs*(returns-values)
    total_loss=(critic_loss+actor_loss).mean()

    return total_loss

  def choose_action(self,node,action):
      state_vector=self.env.state_to_vector(node,action)
      state=T.tensor([state_vector],dtype=T.float)
      pi,v=self.forward(state)
      probs=T.softmax(pi,dim=1)
      dist=Categorical(probs)
      action=dist.sample().numpy()[0]#take a sample from the categorical dist from 1-22
      return action


