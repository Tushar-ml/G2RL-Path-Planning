from map_generator import map_to_value, start_end_points, value_to_map
from global_mapper import find_path, return_path
from PIL import Image
import numpy as np
from dynamic_obstacle import initialize_objects, update_coords

map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
no_of_amr = 20
agent = [1]
# map_path = 'data/random-1.png'
map_img_arr = np.asarray(Image.open(map_path))
coord, inst_arr = initialize_objects(map_img_arr, no_of_amr)

for a in agent:
    inst_arr[coord[a][0], coord[a][1]] = [255,0,0]

value_map = map_to_value(inst_arr)
start_end_coords = start_end_points(coord, value_map)

paths = dict()
for idx, idx_coords in start_end_coords:
    path, fov= find_path(value_map, idx_coords[:2], idx_coords[2:])
    short_path = return_path(path)

    paths[idx] = short_path

# arr = value_to_map(paths, map_img_arr, [], False)
update_coords(paths,inst_arr, 1) # which will update the actions