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

def update_coords(coords, inst_arr,agent, time_idx, width, global_map, direction, agent_old_coordinates):

    h,w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])

    for idx, coord in coords.items():

        if idx == agent:
            agentDone = False
            h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
            h_new, w_new == h_old + direction[0], w_old + direction[1]

            if (h_new == coord[-1][0] and w_new == coord[-1][1]) or (h_new >= h or w_new >= w):
                agentDone = True
            
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

    return np.array(local_obs), np.array(local_map), global_map, agentDone

    
    

    

        




