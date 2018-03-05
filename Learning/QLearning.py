from Environment.soccer import Player,World
from Environment.testbench import create_state_comb,print_status
import numpy as np
import time
from Learning.testPlot import plot
np.random.seed(100)

def construct_env(world):
    world.set_world_size(x=4, y=2)
    player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    player_b = Player(x=1, y=0, has_ball=True, p_id='B')
    world.place_player(player_a, player_id='A')
    world.place_player(player_b, player_id='B')
    world.set_goals(100, 0, 'A')
    world.set_goals(100, 3, 'B')
    playerA_state = int(world.get_state_id(player_a))
    playerB_state = int(world.get_state_id(player_b))
    return playerA_state, playerB_state



def decay(start_value):
    start_value *=0.999985
    return start_value if start_value > 0.001 else 0.001


def run(num_episodes=10):
    q_table_A = np.zeros((len(total_states), num_actions))
    q_table_B = np.zeros((len(total_states), num_actions))
    total_rewards_A = []
    total_rewards_B = []
    q_s_S_A = []
    gamma = 0.9
    epsilon = 1.0
    alpha = 1.0

    for episodes in range(num_episodes):
        episodic_rewards_A = 0.0
        episodic_rewards_B = 0.0
        isGoal = False

        playerA_state, playerB_state = construct_env(world)
        assert playerA_state ==2
        assert playerB_state ==1
        while not isGoal:
            if np.random.rand(1) < epsilon:
                action_num_A = np.random.choice(5)
                action_num_B = np.random.choice(5)

            else:
                action_num_A = np.argmax(q_table_A[playerA_state])
                action_num_B = np.argmax(q_table_B[playerB_state])

            actions = {'A': action_num_A, 'B': action_num_B}
            next_state,rewards,isGoal = world.move(actions)
            # states = [int(k) for s in next_state if s.isdigit() for k in s]
            state_val = total_states[next_state]

            next_playerA_state = state_val

            next_playerB_state = state_val

            if not isGoal:
                max_action_A = int(np.max(q_table_A[next_playerA_state]))
                max_action_B = int(np.max(q_table_B[next_playerB_state]))
            else:
                max_action_A = 0
                max_action_B = 0

            q_table_A[playerA_state,action_num_A] +=alpha * (rewards['A'] +
                                                            gamma * max_action_A - q_table_A[playerA_state,action_num_A])
            q_table_B[playerB_state,action_num_B] += alpha * (rewards['B']+
                                                              gamma *max_action_B-q_table_B[playerB_state,action_num_B])


            playerA_state = next_playerA_state
            playerB_state = next_playerB_state

            episodic_rewards_A += rewards['A']
            episodic_rewards_B +=rewards['B']

        if episodes%2==0:
            epsilon = decay(epsilon)
            alpha = decay(alpha)

        q_s_S_A.append(q_table_A[2,1]) #player A initial state and taking action south
        print("Episode number ", episodes, " Epsilon ", epsilon, " Alpha ", alpha)

        if episodes % 1000 ==0:
            print("Episode:",episodes)
            print(q_table_A[playerA_state,1])

    return total_rewards_A,total_rewards_B,q_s_S_A


if __name__=="__main__":
    num_states = 8
    num_actions = 5

    total_states = create_state_comb(range(num_states), range(num_states))
    world = World()
    reward_A,reward_B, q_Value_sS = run(1000000)
    np.save("qlearning_val_1M_exp",np.array(q_Value_sS))
    plot("qlearning_val_1M_exp.npy","QLearning_plot_Alternate","Q-Learning")









