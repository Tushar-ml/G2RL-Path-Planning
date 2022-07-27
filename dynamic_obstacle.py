import matplotlib.pyplot as plt
from collections import defaultdict
import os
from PIL import Image
from numpy import array, asarray
import numpy as np
import random

img_path = './data/cleaned_empty/empty-48-48-random-1_60_agents.png'



def initialize_objects(arr, n_dynamic_obst = 10):
    
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

def direction(start,final):
    return [final[0]-start[0], final[1]-start[1]]

def update_coords(coords, inst_arr,agent, time_idx, waiting_list, width, global_map):
    
    # images_agent = []
    # images_map = []

    h,w = inst_arr.shape[:2]
    # time_idx = 1
    # images_map.append(Image.fromarray(inst_arr, 'RGB'))

    local_coords = coords[agent][0]

    # images_agent.append(Image.fromarray(np.ones((2*width, 2*width, 3), np.uint8)*255))
    # images_agent.append(Image.fromarray(inst_arr[local_coords[0] - width : local_coords[0]+width,local_coords[1] - width : local_coords[1]+width ]))

    
    local_obs = []
    local_map = np.array([])
    # while time_idx < timestamp:
    for idx, coord in coords.items():
        isEnd = False
        
        if idx in waiting_list and len(waiting_list[idx]) > 0:
                idx_wait_info = waiting_list[idx]
                
                h_new, w_new = list(idx_wait_info.keys())[0]
                local_time_idx = idx_wait_info[(h_new,w_new)]
                try:
                    h_old, w_old = coord[local_time_idx-1]
                except Exception as e:
                    return "Program Failed"
                    
        else:
            if time_idx < len(coord):
                h_old, w_old = coord[time_idx-1]
                h_new, w_new = coord[time_idx]
            
            else:
                h_old, w_old = coord[-1]
                h_new, w_new = coord[-1]
                isEnd = True
        
        # dir_x, dir_y = direction([h_old, w_old],[h_new, w_new])

        if h_new >= h or w_new >= w:
            continue
        cell_coord = inst_arr[h_new, w_new]
        

        if not isEnd and ((cell_coord[0] == 255 and cell_coord[1] == 165 and cell_coord[2] == 0) or (cell_coord[0] == 255 and cell_coord[1] == 0 and cell_coord[2] == 0)):
            if (h_new, w_new) not in waiting_list[idx]:
                waiting_list[idx][(h_new, w_new)] = time_idx
            continue
        
        # inst_arr[h_new, w_new] = inst_arr[h_old, w_old]
        inst_arr[h_new, w_new] = [255,0,0] if idx == agent else [255,165,0]
        
        if not isEnd:
            inst_arr[h_old, w_old] = [255,255,255]

        if idx == agent:
            local_obs = inst_arr[h_new - width:h_new + width, w_new - width:w_new + width]
            global_map[h_old, w_old] = 255
            local_map = global_map[h_new - width:h_new + width, w_new - width:w_new + width]

    return np.array(local_obs), inst_arr, waiting_list, np.array(local_map), global_map

        # time_idx += 1
        # if len(local_obs)>0:
        #     img_agent = Image.fromarray(asarray(local_obs), 'RGB')
        #     images_agent.append(img_agent)

        # if len(inst_arr) > 0:
        #     img_map = Image.fromarray(inst_arr, 'RGB')
        #     images_map.append(img_map)
        

    # images_agent[0].save('data/agent_obs.gif',
    #             save_all=True, append_images=images_agent[1:], optimize=False, duration=timestamp*2, loop=0)
    
    # images_map[0].save('data/dynamic_obst.gif',
    #             save_all=True, append_images=images_map[1:], optimize=False, duration=timestamp*2, loop=0)

    
    

    

        




