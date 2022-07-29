from PIL import Image
import numpy as np
from dynamic_obstacle import initialize_objects, update_coords
from map_generator import start_end_points, map_to_value, global_guidance
from global_mapper import find_path, return_path
from utils import symmetric_pad_array

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

class WarehouseEnvironment:

    def __init__(self,height = 48, width = 48, amr_count = 20, agent_idx = 1, local_fov = 15):

        assert height == 48 and width == 48, "We are not currently supporting other dimensions"
        self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
        self.amr_count = amr_count
        self.map_img_arr = np.asarray(Image.open(self.map_path))
        self.n_states = height*width
        self.n_actions = len(self.action_space())
        self.agent_idx = agent_idx
        self.local_fov = local_fov
        self.time_idx = 1
        self.init_arr = []
    
    def reset(self):
        self.coord, self.init_arr = initialize_objects(self.map_img_arr, self.amr_count)
        
        self.agent_prev_coord = self.coord[self.agent_idx]
        self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255,0,0]

        self.generate_end_points_and_paths()
        self.time_idx = 1
        self.scenes = []
        self.cells_skipped = 0

        reset_state = self.coord[self.agent_idx]
        self.dist = manhattan_distance(reset_state[0], reset_state[1], reset_state[2], reset_state[3])
        graphical_state, _, _,_ = self.step(4)
        return reset_state[0] * reset_state[1], graphical_state
    
    def generate_end_points_and_paths(self):
        value_map = map_to_value(self.init_arr)
        start_end_coords = start_end_points(self.coord, value_map)

        self.agents_paths = dict()
        for idx, idx_coords in start_end_coords:
            path, fov= find_path(value_map, idx_coords[:2], idx_coords[2:])
            short_path = return_path(path)

            self.agents_paths[idx] = short_path

        self.global_mapper_arr = global_guidance(self.agents_paths[self.agent_idx], self.map_img_arr)

    def step(self, action):
        if len(self.init_arr) == 0:
            print("Run env.reset() first")
            return

        self.time_idx += 1
        conv,x,y = self.action_dict[action]
        # print(f'Action taken: {conv}')
        
        target_array = (2*self.local_fov, 2*self.local_fov, 4)
        local_obs, local_map, self.global_mapper_arr, isAgentDone, rewards, self.cells_skipped, self.init_arr, self.agent_prev_coord, self.dist = update_coords(
            self.agents_paths, self.init_arr, self.agent_idx, self.time_idx,
            self.local_fov, self.global_mapper_arr, [x,y], self.agent_prev_coord,
            self.cells_skipped, self.dist
        )

        combined_arr = np.array([])
        if len(local_obs) > 0:
            self.scenes.append(Image.fromarray(local_obs, 'RGB'))
            local_map = local_map.reshape(local_map.shape[0],local_map.shape[1],1)
            combined_arr = np.dstack((local_obs, local_map))
            combined_arr = symmetric_pad_array(combined_arr, target_array, 255)
            combined_arr = combined_arr.reshape(1,1,combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[2])

        return combined_arr, self.agent_prev_coord[0] * self.agent_prev_coord[1], rewards, isAgentDone
    
    def render(self):
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        img = Image.fromarray(self.init_arr,'RGB')
        img.show()

    def create_scenes(self, path = "data/agent_locals.gif", length_s = 100):
        if len(self.scenes) > 0:
            self.scenes[0].save(path,
                 save_all=True, append_images=self.scenes[1:], optimize=False, duration=length_s*4, loop=0)
        else:
            pass
    def action_space(self):
        self.action_dict = {
            0:['up',0,1],
            1:['down',0,-1],
            2:['left',-1,0],
            3:['right',1,0],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())


# import random
# actions = [0,1,2,3,4]


env = WarehouseEnvironment()
_, state = env.reset()

print(state.shape)
# env.render()
# print(coord)

# for ep in range(2):
#     env.reset()
#     print(f'Episode: {ep}')
#     for i in range(100):
#         act = random.choice(actions)
#         new_state,rewards,isDone = env.step(act)
#         print(rewards)
#         if isDone:
#             print("Reached Gole")
#             break


#     env.create_scenes(path = f"data/agent_local_{ep}.gif")


