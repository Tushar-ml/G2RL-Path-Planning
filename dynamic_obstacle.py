from collections import defaultdict
import os
from PIL import Image
from numpy import array, asarray
import numpy as np
import random

def initialize_objects(arr, n_dynamic_obst = 10):
    arr = arr.copy()
    coord = []
    h,w = arr.shape[:2]

    while n_dynamic_obst > 0:
        h_obs = random.randint(0,h-1)
        w_obs = random.randint(0,w-1)

        cell_coord = arr[h_obs, w_obs]
        if cell_coord[0] != 0 and cell_coord[1] != 0 and cell_coord[2] != 0:
            arr[h_obs, w_obs] = [255,165,0]
            n_dynamic_obst -= 1
            coord.append([h_obs,w_obs])
    
    return coord, arr

def update_coords(coords, inst_arr,agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped):

    h,w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0

    for idx, coord in coords.items():

        if idx == agent:
            agentDone = False
            h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
            h_new, w_new == h_old + direction[0], w_old + direction[1]


            if (h_new == coord[-1][0] and w_new == coord[-1][1]) or (h_new >= h or w_new >= w):
                agentDone = True

            if (global_map[h_new, w_new] == 255):
                agent_reward += rewards_dict('0')
                cells_skipped += 1
            
            elif (inst_arr[h_new,w_new][0] == 255 and inst_arr[h_new,w_new][0] == 165 and inst_arr[h_new,w_new][0] == 0) or (inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][0] == 0):
                agent_reward += rewards_dict('1')
                print('Crashed')
                agentDone = True

            elif (global_map[h_new, w_new] != 255 and cells_skipped >= 0):
                agent_reward += rewards_dict('2',cells_skipped)
                cells_skipped = 0
            
            if not agentDone:
                inst_arr[h_new, w_new] = [255,0,0]
                inst_arr[h_old, w_old] = [255,255,255]
                
                if idx == agent:
                    local_obs = inst_arr[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
                    global_map[h_old, w_old] = 255
                    local_map = global_map[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
        
        else:
            isEnd = False
            if time_idx < len(coord):
                h_old, w_old = coord[time_idx-1]
                h_new, w_new = coord[time_idx]
            
            else:
                h_old, w_old = coord[-1]
                h_new, w_new = coord[-1]
                isEnd = True

            if not isEnd:
                inst_arr[h_new, w_new] = [255,165,0]
                inst_arr[h_old, w_old] = [255,255,255]

    return np.array(local_obs), np.array(local_map), global_map, agentDone, agent_reward, cells_skipped


def rewards_dict(case, N = 0):
    r1,r2,r3 = -0.01, -0.1, 0.1
    rewards = {
        '0':r1,
        '1':r1 + r2,
        '2': r1 + N*r3
    }

    return rewards[case]

    
    
    

    

        




