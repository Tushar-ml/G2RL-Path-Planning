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

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist):

    h,w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0
    coord = coords[agent]
    # for idx, coord in coords.items():

    # if idx == agent:
    agentDone = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[0], w_old + direction[1]

    if (h_new == coord[-1][0] and w_new == coord[-1][1]):
        print("Agent Reached Gole")
        agentDone = True

    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0):
        agent_reward += rewards_dict('1')
        agentDone = True

    else:
        if (inst_arr[h_new,w_new][0] == 255 and inst_arr[h_new,w_new][1] == 165 and inst_arr[h_new,w_new][2] == 0) or (inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][1] == 0 and inst_arr[h_new,w_new][2] == 0):
            agent_reward += rewards_dict('1')
            agentDone = True

        if (global_map[h_new, w_new] == 255) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('0')
            cells_skipped += 1
        
        if (global_map[h_new, w_new] != 255 and cells_skipped >= 0) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('2',cells_skipped)
            cells_skipped = 0
    
    if 0 > h_new or h_new>=h or 0>w_new or w_new>= w:
        h_new, w_new = h_old, w_old

    if manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1]) < dist:
        # agent_reward += rewards_dict('3')
        dist = manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1])
    
    inst_arr[h_old, w_old] = [255,255,255]
    inst_arr[h_new, w_new] = [255,0,0]

    # if idx == agent:
    local_obs = inst_arr[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
    
        # else:
        #     isEnd = False
        #     if time_idx < len(coord):
        #         h_old, w_old = coord[time_idx-1]
        #         h_new, w_new = coord[time_idx]
            
        #     else:
        #         h_old, w_old = coord[-1]
        #         h_new, w_new = coord[-1]
        #         isEnd = True

        #     if not isEnd:
        #         inst_arr[h_new, w_new] = [255,165,0]
        #         inst_arr[h_old, w_old] = [255,255,255]
    
    return np.array(local_obs), np.array(local_map), global_map, agentDone, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist


def rewards_dict(case, N = 0):
    r1,r2,r3,r4 = -0.01, -0.1, 0.1, 0.05
    rewards = {
        '0':r1,
        '1':r1 + r2,
        '2': r1 + N*r3,
        '3': r4
    }

    return rewards[case]

    
    
    

    

        




