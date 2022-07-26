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

def update_coords(coords, arr, timestamp = 50, images = []):
    
    
    choices = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]
    new_coord = coords.copy()
    h,w = arr.shape[:2]
    time_idx = 1
    images.append(Image.fromarray(arr, 'RGB'))
    
    while time_idx < timestamp:
        for idx, coord in coords.items():
            
            isEnd = False
            try:
                h_old, w_old = coord[time_idx-1]
                h_new, w_new = coord[time_idx]
            except:
                h_old, w_old = coord[-1]
                h_new, w_new = coord[-1]
                isEnd = True
            # old_value = arr[h_old][w_old]
            
            dir_x, dir_y = direction([h_old, w_old],[h_new, w_new])

            if h_new >= h or w_new >= w:
                continue
            cell_coord = arr[h_new, w_new]
            
            arr[h_new, w_new] = arr[h_old, w_old]
            if not isEnd:
                arr[h_old, w_old] = [255,255,255]
            # new_coord[idx] = (h_new,w_new)
            # print(f"Object {idx} Moved from {[h_old,w_old]} => {[h_new,w_new]}")

        time_idx += 1

        img = Image.fromarray(arr, 'RGB')
        images.append(img)
    # filename = f"sample_results_1/dyn_move_{k}_"+img_path.split("/")[-1]
    # img.save(f'data/dynamic_pos/{filename}')
    # print(f"{filename} saved!")
    images[0].save('data/dynamic_obst.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=timestamp, loop=0)
    # return new_coord, arr, images
    

def initialize_update_dynamic(arr):

    images = []        
    iterations = 100

    coord,arr = initialize_objects(arr, 40)
    for i in range(iterations):
        coord, arr, images = update_coords(coord, arr,i, images)

    images[0].save('data/dynamic_obst.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=iterations, loop=0)

    

        




