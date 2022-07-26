from map_generator import map_to_value, start_end_points, value_to_map
from global_mapper import find_path, return_path
from PIL import Image
import numpy as np

map_path = "data/cleaned_empty/empty-48-48-random-1_60_agents.png"
map_img_arr = np.asarray(Image.open(map_path))

value_map = map_to_value(map_img_arr)
start, end = start_end_points(value_map)

path, fov= find_path(value_map, start, end)
short_path = return_path(path)

value_to_map(short_path, map_img_arr, start, end, fov, False)