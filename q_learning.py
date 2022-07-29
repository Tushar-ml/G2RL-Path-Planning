import pickle
def training():
    import random
    import numpy as np
    from environment import WarehouseEnvironment

    env = WarehouseEnvironment()
    q_table = np.zeros([env.n_states, env.n_actions])
    # Hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.1

    # For plotting metrics
    rewards_window = []
    all_rewards = []

    for i in range(1, 10001):
        state,_ = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space()) # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            _, next_state, reward, done = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward <= -0.1:
                penalties += 1
            
            state = next_state
            epochs += 1

        env.create_scenes("data/agents_q_learning.gif")
        all_rewards.append(reward)

        if i % 100 == 0:
            # env.create_scenes(f"data/agents_locals_q_learning_{i}.gif")
            rewards_window.append(sum(all_rewards[-100:])/100)
            print(f"Episode: {i} reward: {reward}")

    print("Training finished.\n")

    return q_table, rewards_window, all_rewards

q_tbl, r_win, all_r = training()

with open('./models/q_learning_table.pkl','wb') as f:
    pickle.dump(q_tbl, f)

with open('./models/rewards_window.pkl','wb') as f:
    pickle.dump(r_win, f)

with open('./models/all_rewards.pkl','wb') as f:
    pickle.dump(all_r, f)