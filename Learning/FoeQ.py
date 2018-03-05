import numpy as np
from Environment.soccer import Player,World
from Environment.testbench import create_state_comb
from Learning.QLearning import construct_env
from Learning.testPlot import plot
from cvxopt.modeling import variable
from cvxopt.modeling import op
from cvxopt.solvers import options
from cvxopt import matrix,solvers


def decay(start_value):
    start_value *=0.999985
    return start_value if start_value > 0.001 else 0.001


def run(num_episodes=1):
    q_table = np.ones((len(total_states), num_actions, num_actions))
    q_s_S_A = []
    gamma = 0.9
    alpha = 0.7
    for episodes in range(num_episodes):
        isGoal = False
        playerA_state, playerB_state = construct_env(world)
        while not isGoal:
            action_num_A = np.random.choice(5)
            action_num_B = np.random.choice(5)

            actions = {'A': action_num_A, 'B': action_num_B}
            next_state,rewards,isGoal = world.move(actions)

            state_val = total_states[next_state]

            q_transpose = q_table[state_val].T

            q_constr = np.vstack((q_transpose,np.eye(5))) #remove one row for test

            v_col = np.ones(10) #remove for element - these are accounting for equality constraints
            v_col[:5] = -1
            v_col[5:]=0

            v_col=v_col.reshape((q_constr.shape[0],1))
            # print(v_col.shape)
            q_final = np.hstack((v_col,q_constr))

            g = q_final * -1

            G = matrix(g)

            # print (G)
            H = np.zeros(g.shape[0])
            # H[-1]=-1
            h = matrix(H)

            C = np.zeros(6)
            C[0] = -1
            C = matrix(C)

            A = matrix([[0],[1.],[1.],[1.],[1.],[1.]])

            b = matrix(1.)

            result = solvers.lp(C,G,h,A,b,solver='glpk')['x']

            result_vars = result[1:]

            next_player_state = state_val
            if not isGoal:
                max_action = result[0]#np.max(result_vars)

            else:
                max_action = 0

            q_table[playerA_state,action_num_A,action_num_B] +=alpha * (rewards['A'] +
                                                            gamma * max_action - q_table[playerA_state,action_num_A,action_num_B])

            playerA_state = state_val
            playerB_state = state_val

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
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}
    total_states = create_state_comb(range(num_states), range(num_states))
    world = World()
    q_Value_s = run(1000000)
    q_Value_s_arr = np.array(q_Value_s)
    np.save("foe_q_exp", np.array(q_Value_s_arr))
    plot("foe_q_exp.npy", "foe_q_plot", "Foe_q")