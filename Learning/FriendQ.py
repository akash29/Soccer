from Environment.soccer import Player,World
from Environment.testbench import create_state_comb,print_status
import numpy as np
from Learning.QLearning import construct_env
from Learning.testPlot import plot


def decay(start_value):
    start_value -=0.000001
    return start_value if start_value > 0.001 else 0.001

def run(num_episodes=10):
    q_table = np.zeros((len(total_states), num_actions, num_actions))
    q_s_S_A = []
    gamma = 0.9
    alpha = 0.03

    for episodes in range(num_episodes):

        isGoal = False
        playerA_state, playerB_state = construct_env(world)

        while not isGoal:

            action_num_A = np.random.choice(5)

            action_num_B = np.random.choice(5)

            actions = {'A': action_num_A, 'B': action_num_B}

            next_state,rewards,isGoal = world.move(actions)

            state_val = total_states[next_state]

            next_player_state = state_val

            if not isGoal:
                max_action = int(np.max(q_table[next_player_state]))

            else:
                max_action = 0

            q_table[playerA_state,action_num_A,action_num_B] +=alpha * (rewards['A'] +
                                                            gamma * max_action - q_table[playerA_state,action_num_A,action_num_B])

            playerA_state = next_player_state
            playerB_state = next_player_state

        # if episodes%2==0:
        alpha = decay(alpha)

        q_s_S_A.append(q_table[2,1,4])#init state 2 and taking action S and B sticking
        print("Episode number ", episodes, " Alpha ", alpha)

        if episodes % 1000 ==0:
            print("Episode:",episodes)
            print(q_table[playerA_state,1])

    return q_s_S_A


if __name__=="__main__":
    num_states = 8
    num_actions = 5

    total_states = create_state_comb(range(num_states), range(num_states))
    world = World()
    q_Value_s = run(1000000)
    q_Value_s_arr = np.array(q_Value_s)
    values_reduced = q_Value_s_arr[q_Value_s_arr != 0]
    np.save("friendQ_val_1M_exp", np.array(values_reduced))
    plot("friendQ_val_1M_exp.npy", "FriendQ_plot", "Friend-Q")