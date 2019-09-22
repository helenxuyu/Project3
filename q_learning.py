import numpy as np
from cvxopt import matrix, solvers

class SoccerGamePlayers:
    def __init__(self, num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
        self.num_players = num_players
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.table_v = self.initiate_table_v()
        self.table_q = self.initiate_table_q()
        self.table_pi = self.initiate_table_pi()


    def initiate_table_v(self):
        return np.random.rand(self.num_players, self.num_states)

    def initiate_table_q(self):
        pass

    def initiate_table_pi(self):
        pass

    def step(self, actions, reward, current_state, next_state):
        pass

    def selection_function(self, state, table_v, table_q, table_pi):
        pass

    def get_policy(self, player, state):
        pass

    def choose_actions(self, state):
        # return np.random.randint(self.num_actions, size=self.num_players)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions, size=self.num_players)
        else:
            actions = [None] * self.num_players
            for player in range(self.num_players):
                policy = self.table_pi[player, state]
                p = 0
                random_number = np.random.rand()
                for action_idx in range(self.num_actions):
                    p += policy[action_idx]
                    if random_number < p:
                        actions[player] = action_idx
                        break
        return actions

class SoccerGamePlayersCE(SoccerGamePlayers):
    def __init__(self, num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
        super().__init__(num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma)

    def initiate_table_q(self):
        return np.random.rand(self.num_players, self.num_states, self.num_actions, self.num_actions)

    def initiate_table_pi(self):
        return np.ones((self.num_states, self.num_actions, self.num_actions)) / self.num_actions ** 2

    def step(self, actions, reward, current_state, next_state):
        table_v_a = self.table_v[0]
        table_v_b = self.table_v[1]

        table_q_a = self.table_q[0]
        table_q_b = self.table_q[1]

        # update policy
        self.selection_function(next_state, self.table_v, self.table_q, self.table_pi)

        # update table_q for current state
        prev_q_a = table_q_a[current_state, actions[0], actions[1]]
        prev_q_b = table_q_b[current_state, actions[1], actions[0]]
        table_q_a[current_state, actions[0], actions[1]] = (1 - self.alpha) * table_q_a[current_state, actions[0], actions[1]] +\
                                                           self.alpha * ((1 - self.gamma) * reward + self.gamma * table_v_a[next_state])
        table_q_b[current_state, actions[1], actions[0]] = (1 - self.alpha) * table_q_b[current_state, actions[1], actions[0]] +\
                                                           self.alpha * ((1 - self.gamma) * (-1 * reward) + self.gamma * table_v_b[next_state])
        post_q_a = table_q_a[current_state, actions[0], actions[1]]
        post_q_b = table_q_b[current_state, actions[1], actions[0]]

        # update alpha and epsilon
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return (prev_q_a, post_q_a), (prev_q_b, post_q_b)

    def selection_function(self, state, table_v, table_q, table_pi):
        num_actions = self.num_actions
        num_actions_squared = num_actions ** 2
        table_q_a_state = table_q[0, state]
        table_q_b_state = table_q[1, state]

        c = (table_q_a_state + table_q_b_state).flatten()

        G = np.zeros(((num_actions - 1) * num_actions * 2, num_actions_squared))
        G_row_idx = 0
        G_col_idx = 0
        next_G_col_idx = num_actions
        for i_pivot in range(num_actions):
            for i in range(num_actions):
                if i != i_pivot:
                    G[G_row_idx, G_col_idx:next_G_col_idx] = table_q_a_state[i] - table_q_a_state[i_pivot]
                    G_row_idx += 1
            G_col_idx = next_G_col_idx
            next_G_col_idx += num_actions

        G_col_idx = 0
        for j_pivot in range(num_actions):
            for j in range(num_actions):
                if j != j_pivot:
                    G[G_row_idx, range(G_col_idx, num_actions_squared, num_actions)] = table_q_b_state[:, j] - table_q_b_state[:, j_pivot]
                    G_row_idx += 1
            G_col_idx += 1
        G = np.vstack((-np.eye(num_actions_squared, num_actions_squared), G))

        h = np.zeros(G.shape[0])
        A = np.ones((1, num_actions_squared))
        b = [1.]

        res = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        x = np.array(res['x']).flatten()
        table_pi[state] = x.reshape((num_actions, num_actions))

        table_v[0, state] = np.sum(table_pi[state] * table_q_a_state)
        table_v[1, state] = np.sum(table_pi[state] * table_q_b_state)

    def get_policy(self, player, state):
        return self.table_pi[state].flatten()

    def choose_actions(self, state):
        # return np.random.randint(self.num_actions, size=self.num_players)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions, size=self.num_players)
        else:
            actions = [None] * self.num_players
            for player in range(self.num_players):
                policy = self.table_pi[state]
                p = 0
                random_number = np.random.rand()
                for action_idx_i in range(self.num_actions):
                    for action_idx_j in range(self.num_actions):
                        p += policy[action_idx_i][action_idx_j]
                        if random_number < p:
                            actions=[action_idx_i, action_idx_j]
        return actions

class SoccerGamePlayersFriendQ(SoccerGamePlayers):
    def __init__(self, num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
        super().__init__(num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma)

    def initiate_table_q(self):
        return np.random.rand(self.num_players, self.num_states, self.num_actions, self.num_actions)

    def initiate_table_pi(self):
        return np.ones((self.num_players, self.num_states, self.num_actions)) / self.num_actions

    def step(self, actions, reward, current_state, next_state):
        prev_q_a, post_q_a = self.step_single_player(0, actions, reward, current_state, next_state)
        prev_q_b, post_q_b = self.step_single_player(1, [actions[1], actions[0]], -reward, current_state, next_state)
        return (prev_q_a, post_q_a), (prev_q_b, post_q_b)

    def step_single_player(self, player, actions, reward, current_state, next_state):
        table_v = self.table_v[player]
        table_q = self.table_q[player]
        table_pi = self.table_pi[player]

        # update policy
        self.selection_function(next_state, table_v, table_q, table_pi)

        # update table_q for current state
        prev_q = table_q[current_state, actions[0], actions[1]]
        table_q[current_state, actions[0], actions[1]] = (1 - self.alpha) * table_q[current_state, actions[0], actions[1]] + \
                                                          self.alpha * ((1 - self.gamma) * reward + self.gamma * table_v[next_state])
        post_q = table_q[current_state, actions[0], actions[1]]

        # update alpha and epsilon
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return prev_q, post_q

    def selection_function(self, state, table_v, table_q, table_pi):
        table_q_s = table_q[state]

        # Indexes of the maximal elements of a N-dimensional array
        coordinate = np.unravel_index(np.argmax(table_q_s, axis=None), table_q_s.shape)

        # update table_v at state
        table_v[state] = table_q_s[coordinate]

        # update policy
        table_pi[state] = 0
        table_pi[state, coordinate[0]] = 1


    def get_policy(self, player, state):
        return self.table_pi[player][state]

class SoccerGamePlayersFoeQ(SoccerGamePlayers):
    def __init__(self, num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
        super().__init__(num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma)

    def initiate_table_q(self):
        return np.random.rand(self.num_players, self.num_states, self.num_actions, self.num_actions)

    def initiate_table_pi(self):
        return np.ones((self.num_players, self.num_states, self.num_actions)) / self.num_actions

    def step(self, actions, reward, current_state, next_state):
        prev_q_a, post_q_a = self.step_single_player(0, actions, reward, current_state, next_state)
        prev_q_b, post_q_b = self.step_single_player(1, [actions[1], actions[0]], -reward, current_state, next_state)
        return (prev_q_a, post_q_a), (prev_q_b, post_q_b)

    def step_single_player(self, player, actions, reward, current_state, next_state):
        table_v = self.table_v[player]
        table_q = self.table_q[player]
        table_pi = self.table_pi[player]

        # update policy
        self.selection_function(next_state, table_v, table_q, table_pi)

        # update table_q for current state
        prev_q = table_q[current_state, actions[0], actions[1]]
        table_q[current_state, actions[0], actions[1]] = (1 - self.alpha) * table_q[current_state, actions[0], actions[1]] + \
                                          self.alpha * ((1 - self.gamma) * reward + self.gamma * table_v[next_state])
        post_q = table_q[current_state, actions[0], actions[1]]

        # update alpha and epsilon
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return prev_q, post_q

    def selection_function(self, state, table_v, table_q, table_pi):
        c = np.zeros(self.num_actions + 1)
        c[0] = -1.0
        G = np.empty((self.num_actions, self.num_actions + 1))
        G[:, 1:] = -table_q[state].T
        G[:, 0] = 1
        G = np.vstack((-np.eye(self.num_actions + 1, self.num_actions + 1), G))
        G[0, 0] = 0

        h = np.zeros(self.num_actions + self.num_actions + 1)
        A = np.ones((1, self.num_actions + 1))
        A[0, 0] = 0
        b = [1.0]

        res = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        x = np.array(res['x']).flatten()
        table_pi[state] = x[1:]
        table_v[state] = x[0]

    def get_policy(self, player, state):
        return self.table_pi[player][state]

class SoccerGamePlayersQ(SoccerGamePlayers):
    def __init__(self, num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma):
        super().__init__(num_players, num_states, num_actions, epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, alpha_min, gamma)

    def initiate_table_q(self):
        return np.random.rand(self.num_players, self.num_states, self.num_actions)

    def initiate_table_pi(self):
        return np.ones((self.num_players, self.num_states, self.num_actions)) / self.num_actions

    def step(self, actions, reward, current_state, next_state):
        prev_q_a, post_q_a = self.step_single_player(0, actions[0], reward, current_state, next_state)
        prev_q_b, post_q_b = self.step_single_player(1, actions[1], -reward, current_state, next_state)
        return (prev_q_a, post_q_a), (prev_q_b, post_q_b)

    def step_single_player(self, player, action, reward, current_state, next_state):
        table_v = self.table_v[player]
        table_q = self.table_q[player]
        table_pi = self.table_pi[player]

        # update policy
        self.selection_function(next_state, table_v, table_q, table_pi)

        # update table_q for current state
        prev_q = table_q[current_state, action]
        table_q[current_state, action] = (1 - self.alpha) * table_q[current_state, action] + \
                                         self.alpha * ((1 - self.gamma) * reward + self.gamma * table_v[next_state])
        post_q = table_q[current_state, action]

        # update alpha and epsilon
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return prev_q, post_q

    def selection_function(self, state, table_v, table_q, table_pi):
        table_q_s = table_q[state]
        max_index = np.argmax(table_q_s)
        table_v[state] = table_q_s[max_index]
        table_pi[state] = 0
        table_pi[state, max_index] = 1

    def get_policy(self, player, state):
        return self.table_pi[player][state]