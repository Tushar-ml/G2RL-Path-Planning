import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

def random_map(w,h, n_static, map_name = "random-1", color_coord = [50,205,50]):

    static_coord_width = [random.randint(0,w-1) for i in range(n_static)]
    static_coord_height = [random.randint(0,h-1) for i in range(n_static)]

    data = np.ones((h, w, 3), dtype=np.uint8)*255

    for i in range(n_static):
        data[static_coord_height[i], static_coord_width[i]] = color_coord
    
    img = Image.fromarray(data, 'RGB')
    img.save(f'data/{map_name}.png')
    img.show()


def guide_map(w,h,h_coord,w_coord, map_name = "guide-1", color_coord = [50,205,50]):

    assert len(h_coord) == len(w_coord), "Coordinates length is not same"
    data = np.ones((h, w, 3), dtype=np.uint8)*255

    for i in range(len(h_coord)):
        data[h_coord[i], w_coord[i]] = color_coord
    
    img = Image.fromarray(data, 'RGB')
    img.save(f'data/{map_name}.png')
    img.show()

def map_to_value(arr):
    h,w = arr.shape[:2]
    new_arr = np.zeros(shape=(h,w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            cell_coord = arr[i,j]
            if cell_coord[0] == 0 and cell_coord[1] == 0 and cell_coord[2] == 0:
                new_arr[i,j] = 1

    return new_arr

def start_end_points(arr):
    coords = []

    h,w = arr.shape[:2]
    for i in range(2):
        while True:
            h_new = random.randint(0,h-1)
            w_new = random.randint(0,w-1)

            if arr[h_new][w_new] == 0:
                coords.append([h_new, w_new])
                break

    return coords

def value_to_map(paths, arr, start, end, expands, show_view = False):
    if show_view:
        h,w = len(expands), len(expands[0])
        for i in range(h):
            for j in range(w):
                if expands[i][j] != -1:
                    arr[i,j] = [0,255,0]
    for path_coord in paths:
        x,y = path_coord
        arr[x,y] = [105,105,105]

    arr[start[0],start[1]] = [255,0,0]
    arr[end[0], end[1]] = [255,0,0]

    img = Image.fromarray(arr, "RGB")
    img.save(f'data/a_star_{start[0]}-{start[1]}_{end[0]}-{end[1]}.png')
    img.show()

def heuristic_generator(arr, end):
    try:
        h,w = arr.shape
    except:
        h,w = len(arr), len(arr[0])
    h_map = [[0 for i in range(w)] for j in range(h)]
    for i in range(h):
        for j in range(w):
            h_map[i][j] = abs(end[0] - i) + abs(end[1] - j)

    return h_map

if __name__ == "__main__":
    random_map(100,100, 3000,"random_2")

