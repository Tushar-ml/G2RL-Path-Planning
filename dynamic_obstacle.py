from multiprocessing.spawn import import_main_path
import matplotlib.pyplot as plt

import os
from PIL import Image
from numpy import array, asarray
import random

img_path = './data/cleaned_empty/empty-48-48-random-10_60_agents.png'

images = []


def initialize_objects(img_path, n_dynamic_obst = 10):
    
    img = Image.open(img_path)
    arr = asarray(img)
    coord = []
    h,w = arr.shape[:2]

    while n_dynamic_obst > 0:
        h_obs = random.randint(0,h-1)
        w_obs = random.randint(0,w-1)

        cell_coord = arr[h_obs, w_obs]
        if cell_coord[0] == 255 and cell_coord[1] == 255 and cell_coord[2] == 255:
            arr[h_obs, w_obs] = [255,165,0]
            n_dynamic_obst -= 1
            coord.append((h_obs,w_obs))
            print(f"Obstacle {n_dynamic_obst} placed")

    img = Image.fromarray(arr, 'RGB')
    filename = "sample_results_1/dyn_init_"+img_path.split("/")[-1]
    img.save(f'data/dynamic_pos/{filename}')

    return coord, arr

def update_coords(coords, arr, k):
    
    
    choices = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]
    new_coord = coords.copy()
    h,w = arr.shape[:2]
    for idx, coord in enumerate(coords):
        while True:
            h_old, w_old = coord
            
            choice = random.choice(choices)
            dir_x, dir_y = choice
            h_new, w_new = h_old + dir_x, w_old + dir_y

            if h_new >= h or w_new >= w:
                continue
            cell_coord = arr[h_new, w_new]
            if cell_coord[0] == 255 and cell_coord[1] == 255 and cell_coord[2] == 255:
                arr[h_new, w_new] = [255,165,0]
                arr[h_old, w_old] = [255,255,255]
                new_coord[idx] = (h_new,w_new)
                # print(f"Object {idx} Moved from {[h_old,w_old]} => {[h_new,w_new]}")
                break

    img = Image.fromarray(arr, 'RGB')
    images.append(img)
    filename = f"sample_results_1/dyn_move_{k}_"+img_path.split("/")[-1]
    img.save(f'data/dynamic_pos/{filename}')
    # print(f"{filename} saved!")
    return new_coord, arr
    

                

iterations = 100

coord,arr = initialize_objects(img_path, 20)
for i in range(iterations):
    coord, arr = update_coords(coord, arr,i)

images[0].save('data/dynamic_obst.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=iterations//2, loop=0)

    

        




