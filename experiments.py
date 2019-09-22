from soccer_game import SoccerGame
from q_learning import *
import numpy as np
from cvxopt import solvers
import matplotlib.pyplot as plt
import pandas as pd

def plot_saved_file(filename):
    data = pd.read_csv(filename + ".tsv", sep='\t', header=None)
    data.rename(columns={0:'iter_index', 1:'diff'}, inplace=True)
    data['iter_index'] = data['iter_index'] / 100000
    plot = data.plot(x='iter_index', y='diff', linewidth=0.5, legend=None)
    plot.set_xlabel("Simulation Iteration 10^5")
    plot.set_ylabel("Q-value difference")
    plt.ylim([0, 0.5])
    plt.title(filename[17:-8])
    plt.xticks(np.arange(0, 11, 1.0))
    plt.savefig(filename + ".png")

def save_result_to_file(stats, learning_class, num_iterations):
    file_name = "{0}_{1}.tsv".format(learning_class, num_iterations)
    with open(file_name, 'w') as f:
        for stat in stats:
            f.write(str(stat[0]) + "\t" + str(stat[1]) + '\n')
    f.close()

def run_experiments(learning_class, num_iterations, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
    # create a SoccerGame environment
    env = SoccerGame()
    solvers.options['show_progress'] = False

    # create the learning agent
    learning_agent = learning_class(num_players=2, num_states=env.state_count, num_actions=env.action_count,
                                    epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                                    alpha=alpha, alpha_decay=alpha_decay, alpha_min=alpha_min, gamma=gamma)

    # stats
    stats = []
    iter_index = 0
    episode_count = 0

    # for iter_index in range(num_iterations):
    #     if iter_index % 1000 == 0:
    #         print("iter_{0}".format(iter_index))
    while iter_index < num_iterations:
        env.reset()

        # start to interact with the environment
        game_ends = False
        while not game_ends:
            current_position, current_ball_owner, current_state_index = env.get_current_state()
            actions = learning_agent.choose_actions(current_state_index)
            moving_player = np.random.randint(learning_agent.num_players)

            # step the players in the game
            next_state, reward, done = env.step(actions, moving_player)
            next_position, next_ball_owner, next_state_index = next_state

            qa, qb = learning_agent.step(actions, reward, current_state_index, next_state_index)

            if current_state_index == 71 and actions[0] == 1 and actions[1] == 4:
                qa_diff = abs(qa[1] - qa[0])
                qb_diff = abs(qb[1] - qb[0])
                stats.append([iter_index, qa_diff, qb_diff])

            # stop the game if it is done
            if done:
                game_ends = True

            if iter_index % 10000 == 0:
                print("iter_{0}".format(iter_index))
            iter_index += 1
            if iter_index >= num_iterations:
                break

        episode_count += 1
    # save the stats for future analysis
    print(episode_count)
    save_result_to_file(stats, learning_class.__name__, num_iterations)
    plot_saved_file("{0}_{1}".format(learning_class.__name__, num_iterations))

# CeQ
# run_experiments(learning_class=SoccerGamePlayersCE,
#                 num_iterations = 100000,
#                 epsilon=1.0,
#                 epsilon_decay=1.0,
#                 epsilon_min=0.001,
#                 alpha=0.5,
#                 alpha_decay=0.99999,
#                 alpha_min=0.001,
#                 gamma=0.9)

# FoeQ
run_experiments(learning_class=SoccerGamePlayersFoeQ,
                num_iterations = 100000,
                epsilon=1.0,
                epsilon_decay=1.0,
                epsilon_min=0.001,
                alpha=0.5,
                alpha_decay=0.999998,
                alpha_min=0.001,
                gamma=0.9)

# FriendQ
# run_experiments(learning_class=SoccerGamePlayersFriendQ,
#                 num_iterations = 1000000,
#                 epsilon=1.0,
#                 epsilon_decay=1.0,
#                 epsilon_min=0.001,
#                 alpha=0.5,
#                 alpha_decay=0.99998,
#                 alpha_min=0.001,
#                 gamma=0.9)

# Q-Learning
# run_experiments(learning_class=SoccerGamePlayersQ,
#                 num_iterations = 1000000,
#                 epsilon=1.0,
#                 epsilon_decay=1,
#                 epsilon_min=0.001,
#                 alpha=0.5,
#                 alpha_decay=0.999998,
#                 alpha_min=0.001,
#                 gamma=0.9)