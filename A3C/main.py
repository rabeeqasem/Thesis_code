
import json
import networkx as nx

import torch.optim as optim
import pyproj
import torch.multiprocessing as mp
from environment_f import environment
from gather_eps import gather_eps
from SharedAdam_f import SharedAdam
from Agent import Agent
from ActorCritics_f import ActorCritics

if __name__ == '__main__':

  def get_neighbors(node):
    neighbors=[i for i in h.neighbors(node)]
    return len(neighbors)


  f = open('final.json')

  data_str_key=json.load(f)

  data={}
  for k,v in data_str_key.items():
    data[int(k)]=v

  #reconstruct the graph
  h = nx.Graph()
  for key in data.keys():
    if data[key]['type']=='way':
      for i in range(len(data[key]['nodes'])-1):
        if 'tags' in data[key] and 'name' in data[key]['tags']:
            h.add_edge(data[key]['nodes'][i],data[key]['nodes'][i+1],parent=data[key]['id'],label=data[key]['tags']['name'])
        else:
          h.add_edge(data[key]['nodes'][i],data[key]['nodes'][i+1],parent=data[key]['id'])

  geod = pyproj.Geod(ellps='WGS84')

  # Compute distance among the two nodes indexed[s] indexed[d] using LON and LAT
  for s,d in h.edges():
    azimuth1, azimuth2, distance = geod.inv(data[s]['lon'],data[s]['lat'],data[d]['lon'],data[d]['lat'])
    h.edges[s,d]['weight'] = distance

  nodex={}
  for node in h.nodes:
    nodex[node]=get_neighbors(node)

  mx = max(nodex.values())
  [k for k, v in nodex.items() if v == mx]

  t=[i for i in h.neighbors(2003461246)]
  t.append(2003461246)

  node_list=[2003461246]
  for node in h.neighbors(2003461246):
    node_list.append(node)
    for snode in h.neighbors(node):
      node_list.append(snode)

  dictt={}
  for node in node_list:
    c=0
    for subnode in node_list:
      if subnode in h[node]:
        c+=1
    dictt[node]=c

  g = h.subgraph(dictt.keys())

  lr=1e-2
  n_actions=len(g.nodes)
  input=len(g.nodes)*2
  N_games=1000
  T_max=5
  env=environment(g)
  gather_info=gather_eps()



  global_actor_critic=ActorCritics(input,n_actions,env=env)
  global_actor_critic.share_memory()
  optim=SharedAdam(global_actor_critic.parameters(),lr=lr,betas=(0.92,0.999))
  global_ep=mp.Value('i',0)
  res_queue=mp.Queue()###
  #cpu_count=mp.cpu_count()
  cpu_count=4
  workers=[Agent(global_actor_critic,optim,input,n_actions,gamma=0.99,lr=lr,worker_name=i,
    global_episode_index=global_ep,gather=gather_info,env=env,games=N_games,T_max=T_max) for i in range(cpu_count)]
  print('CPU count=',cpu_count)
  print('start')
  [w.start() for w in workers]
  res = []
  while True:
    r = res_queue.get()
    if r is not None:
      res.append(r)
    else:
      break
  print('join',flush=True)
  [w.join() for w in workers]
  gather_info.print_dict()



  print('---------------------------------------------------------------')
  obs=env.reset()
  soruce,end=env.state_dec(obs)
  result=[env.dec_node[soruce]]
  done=False
  itr=0
  while not done:
    soruce,end=env.state_dec(obs)
    action=global_actor_critic.choose_action(soruce,end)
    obs_,reward,done=env.step(obs,action)
    obs=obs_
    soruce,end=env.state_dec(obs)
    nnode=env.dec_node[soruce]
    result.append(nnode)
    itr+=1


  print('shortest path is ',result)
  