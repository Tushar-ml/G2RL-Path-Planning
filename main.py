from map_generator import map_to_value, start_end_points, global_guidance, local_guidance
from global_mapper import find_path, return_path
from PIL import Image
import numpy as np
from dynamic_obstacle import initialize_objects, update_coords
from collections import defaultdict
from cnn_arch import get_cnn_model
from utils import symmetric_pad_array

map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
no_of_amr = 20
agent = 1
# map_path = 'data/random-1.png'
map_img_arr = np.asarray(Image.open(map_path))
coord, inst_arr = initialize_objects(map_img_arr, no_of_amr)


inst_arr[coord[agent][0], coord[agent][1]] = [255,0,0]

value_map = map_to_value(inst_arr)
start_end_coords = start_end_points(coord, value_map)

paths = dict()
for idx, idx_coords in start_end_coords:
    path, fov= find_path(value_map, idx_coords[:2], idx_coords[2:])
    short_path = return_path(path)

    paths[idx] = short_path

global_mapper_arr = global_guidance(paths[agent], map_img_arr)
global_mapper_arr = np.asarray(global_mapper_arr)
gmap = Image.fromarray(global_mapper_arr,"L")
gmap.save("data/global_agent_map.png")
############ Update Movement of All AMRs #############
h,w = 48,48
nt = 4
timestamp = 50
time_idx = 1
waiting_list = defaultdict(dict)

images_map = []
images_agent = []
images_local_map = []

time_idx = 1
width = 15
images_map.append(Image.fromarray(inst_arr, 'RGB'))
images_local_map.append(Image.fromarray(np.ones((2*width, 2*width), np.uint8)*255))

local_coords = paths[agent][0]

# images_agent.append(Image.fromarray(inst_arr[max(0,local_coords[0] - width) : min(h-1,local_coords[0]+width),max(0,local_coords[1] - width) : min(w-1,local_coords[1]+width) ]))
# agent_map_local = np.asarray([])
target_array = (2*width, 2*width, 4)
model_3d_cnn = get_cnn_model(2*width, 2*width, 4,nt)

input_arr = []
while time_idx < timestamp:
    
    local_obs, local_map, global_mapper_arr = update_coords(paths,inst_arr, 1,time_idx, width,global_mapper_arr)
    # if len(local_obs)>0:
    #         img_agent = Image.fromarray(np.asarray(local_obs), 'RGB')
    #         images_agent.append(img_agent)

    # if len(inst_arr) > 0:
    #     img_map = Image.fromarray(inst_arr, 'RGB')
    #     images_map.append(img_map)

    # if len(local_map) > 0:
    #     images_local_map.append(Image.fromarray(local_map,'P'))
    if len(local_map) > 0 and len(local_obs) > 0:
        local_map = local_map.reshape(local_map.shape[0],local_map.shape[1],1)
        combined_arr = np.dstack((local_obs, local_map))
        combined_arr = symmetric_pad_array(combined_arr, target_array, 255).tolist()
        input_arr.append(combined_arr)

        if len(input_arr) == nt:
            final_input_arr = np.array(input_arr)
            # print(final_input_arr.shape)
            final_input_arr = final_input_arr.reshape(1, nt, final_input_arr.shape[1], final_input_arr.shape[2], final_input_arr.shape[0])
            out = model_3d_cnn.predict(final_input_arr)
            print(out)
            input_arr.pop(0)
 
        
        

    time_idx += 1




# model_3d_cnn.fit(inputs)

# images_agent[0].save('data/agent_obs.gif',
#                 save_all=True, append_images=images_agent[1:], optimize=False, duration=timestamp*4, loop=0)
    
# images_map[0].save('data/dynamic_obst.gif',
#             save_all=True, append_images=images_map[1:], optimize=False, duration=timestamp*4, loop=0)  



# images_local_map[0].save('data/agent_local_map.gif',
#                 save_all=True, append_images=images_local_map[1:], optimize=False, duration=timestamp*4, loop=0)