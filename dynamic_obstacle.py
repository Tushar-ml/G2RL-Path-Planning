from multiprocessing.spawn import import_main_path
import matplotlib.pyplot as plt

import os
from PIL import Image
from numpy import array, asarray
import random

img_path = './data/cleaned_empty/empty-48-48-random-10_60_agents.png'



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
            # print(f"Obstacle {n_dynamic_obst} placed")

    # img = Image.fromarray(arr, 'RGB')
    # filename = "sample_results_1/dyn_init_"+img_path.split("/")[-1]
    # img.save(f'data/dynamic_pos/{filename}')

    return coord, arr

def direction(start,final):
    return [final[0]-start[0], final[1]-start[1]]

def update_coords(coords, inst_arr,agent, timestamp = 50):
    
    images_agent = []
    images_map = []
    width = 4
    choices = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]
    new_coord = coords.copy()
    h,w = inst_arr.shape[:2]
    time_idx = 1
    images_map.append(Image.fromarray(inst_arr, 'RGB'))

    local_coords = coords[agent][0]
    images_agent.append(Image.fromarray(inst_arr[local_coords[0] - 4 : local_coords[0]+4,local_coords[1] - 4 : local_coords[1]+4 ]))

    while time_idx < timestamp:
        local_obs = []
        for idx, coord in coords.items():
            
            isEnd = False
            try:
                h_old, w_old = coord[time_idx-1]
                h_new, w_new = coord[time_idx]
            except:
                h_old, w_old = coord[-1]
                h_new, w_new = coord[-1]
                isEnd = True
            # old_value = inst_arr[h_old][w_old]
            
            dir_x, dir_y = direction([h_old, w_old],[h_new, w_new])

            if h_new >= h or w_new >= w:
                continue
            cell_coord = inst_arr[h_new, w_new]
            
            inst_arr[h_new, w_new] = inst_arr[h_old, w_old]
            if not isEnd:
                inst_arr[h_old, w_old] = [255,255,255]

            if idx == agent:
                local_obs = inst_arr[h_new - 4:h_new + 4, w_new - 4:w_new + 4]
                    
            # new_coord[idx] = (h_new,w_new)
            # print(f"Object {idx} Moved from {[h_old,w_old]} => {[h_new,w_new]}")

        time_idx += 1
        img_agent = Image.fromarray(local_obs, 'RGB')
        img_map = Image.fromarray(inst_arr, 'RGB')

        images_map.append(img_map)
        images_agent.append(img_agent)
    # filename = f"sample_results_1/dyn_move_{k}_"+img_path.split("/")[-1]
    # img.save(f'data/dynamic_pos/{filename}')
    # print(f"{filename} saved!")
    
    images_agent[0].save('data/agent_obs.gif',
                save_all=True, append_images=images_agent[1:], optimize=False, duration=timestamp*2, loop=0)
    
    images_map[0].save('data/dynamic_obst.gif',
                save_all=True, append_images=images_map[1:], optimize=False, duration=timestamp*2, loop=0)
    

    

        




