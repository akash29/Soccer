import numpy as np
from Environment.soccer import Player,World
from Environment.testbench import create_state_comb
from Learning.QLearning import construct_env
from Learning.testPlot import plot

from cvxopt import matrix,solvers

def decay(start_value):
    start_value *=0.999985
    return start_value if start_value > 0.001 else 0.001


def run(num_episodes=1):
    q_table_A = np.ones((len(total_states), num_actions, num_actions))
    q_table_B = np.ones((len(total_states), num_actions, num_actions))
    q_s_S_A = []
    gamma = 0.9
    alpha = 0.8
    for episodes in range(num_episodes):
        isGoal = False
        playerA_state, playerB_state = construct_env(world)
        while not isGoal:
            action_num_A = np.random.choice(5)
            action_num_B = np.random.choice(5)

            actions = {'A': action_num_A, 'B': action_num_B}
            next_state,rewards,isGoal = world.move(actions)

            state_val = total_states[next_state]

            #>=0 constraint
            constr1 = np.eye(25)

            # rationality constraints
            q1_A = q_table_A[state_val]
            q_lot=[]
            for i in range(num_actions):
                q_temp = []
                for j in range(num_actions):
                    if i!=j:
                        q_temp.append(q1_A[i,:]-q1_A[j,:])

                q_lot.append(q_temp)


            player_1_constr = np.zeros((20,25))
            player_1_constr[:4,:5] = q_lot[0]
            player_1_constr[4:8,5:10] = q_lot[1]
            player_1_constr[8:12,10:15] = q_lot[2]
            player_1_constr[12:16, 15:20] = q_lot[3]
            player_1_constr[16:20,20:25] = q_lot[4]

            q1_B = q_table_B[state_val]
            q_lot = []
            for j in range(num_actions):
                q_temp = []
                for i in range(num_actions):
                    if i != j:
                        temp = q1_B[:, j] - q1_B[:, i]
                        q_temp.append(temp.T)
                q_lot.append(q_temp)

            player_2_constr = np.zeros((20, 25))
            player_2_constr[:4, :5] = q_lot[0]
            player_2_constr[4:8, 5:10] = q_lot[1]
            player_2_constr[8:12, 10:15] = q_lot[2]
            player_2_constr[12:16, 15:20] = q_lot[3]
            player_2_constr[16:20, 20:25] = q_lot[4]

            player_constraints = np.vstack((constr1,player_1_constr,player_2_constr))


            G = player_constraints*-1

            G = matrix(G)

            # C = np.zeros(25)

            C = np.add(q_table_A[state_val],q_table_B[state_val]).reshape(1,25)

            C *=-1

            C = matrix(C[0])

            h = np.zeros(player_constraints.shape[0])

            h = matrix(h)

            A = np.ones((1,25))

            A = matrix(A)

            b = matrix(1.)

            result = solvers.lp(C,G,h,A,b)['x']

            if result !=None:

                q_A_flatten = np.array(q_table_A[state_val]).flatten().reshape(25,1)

                q_B_flatten = np.array(q_table_B[state_val]).flatten().reshape(25,1)

                # result = np.zeros((25, 1))

                max_A = result * q_B_flatten

                max_B = result * q_A_flatten

                result_A = sum(max_A)

                result_B = sum(max_B)

                # result_A = np.dot(result_x,q_A_flatten)
                #
                # result_B = np.dot(result_x,q_B_flatten)

                if not isGoal:
                    max_action_A = result_A  # np.max(result_vars)
                    max_action_B = result_B

                else:
                    max_action_A = 0
                    max_action_B = 0

                q_table_A[playerA_state, action_num_A, action_num_B] += alpha * (rewards['A'] +
                                                                                 gamma * max_action_A - q_table_A[
                                                                                     playerA_state, action_num_A, action_num_B])

                q_table_B[playerB_state, action_num_A, action_num_B] += alpha * (rewards['B'] +
                                                                                 gamma * max_action_B - q_table_B[
                                                                                     playerB_state, action_num_A, action_num_B])
                playerA_state = state_val
                playerB_state = state_val

        # if episodes%2==0:
        alpha = decay(alpha)

        q_s_S_A.append(q_table_A[2, 1, 4])  # init state 2 and taking action S and B sticking
        print("Episode number ", episodes, " Alpha ", alpha)


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
    np.save("ce_q_exp", np.array(q_Value_s_arr))
    plot("ce_q_exp.npy", "ce_q_plot", "ce_q")