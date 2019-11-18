from numpy.random import choice

NUM_ROWS = 2
NUM_COLS = 4
NUM_PLAYERS = 2
ACTION_COUNT = 5

ACTION_N = 0
ACTION_S = 1
ACTION_E = 2
ACTION_W = 3
ACTION_STICK = 4

class SoccerGame:

    def __init__(self, positions=[[0, 2],[0, 1]], ball_owner=1):
        self.positions = positions
        self.ball_owner = ball_owner
        self.action_count = ACTION_COUNT
        self.state_count = NUM_ROWS * NUM_COLS * (NUM_ROWS * NUM_COLS - 1) * 2
        self.valid_initial_positions = [[i, j] for i in range(NUM_ROWS) for j in range(1,NUM_COLS-1)]

    # reset to a random start position and ball_owner
    def reset(self):
        # rather than a random state, reset to a specific state
        positions = choice(len(self.valid_initial_positions), 2, replace=False)
        self.positions = [self.valid_initial_positions[positions[0]], self.valid_initial_positions[positions[1]]]
        self.ball_owner = choice([0, 1])
        # self.positions = [[0, 2], [0, 1]]
        # self.ball_owner = 1


    # return [next_state, reward, done] information
    def step(self, actions, moving_player):
        reward = self.move_player(actions, moving_player)
        if reward != 0:
            return self.get_current_state(), reward, reward != 0
        else:
            reward = self.move_player(actions, 1 - moving_player)
            return self.get_current_state(), reward, reward != 0

    def move_player(self, actions, moving_player):
        position = self.positions[moving_player].copy()
        action = actions[moving_player]
        if action == ACTION_N:
            if position[0] > 0:
                position[0] -= 1
        elif action == ACTION_S:
            if position[0] < NUM_ROWS - 1:
                position[0] += 1
        elif action == ACTION_E:
            if position[1] < NUM_COLS - 1:
                position[1] += 1
        elif action == ACTION_W:
            if position[1] > 0:
                position[1] -= 1

        # moves into the stationary player, move does not take place
        # the possession of the ball changes
        if position == self.positions[1-moving_player]:
            self.ball_owner = 1 - moving_player
            return 0
        else:
            # update the position of the moving player
            self.positions[moving_player] = position
            if moving_player == self.ball_owner:
                return self.check_goal(self.positions[moving_player])
            else:
                return 0

    def check_goal(self, position):
        # whoever the player is, when col == 0, A will get 100
        # when col == NUM_COLS - 1, A will get -100
        # All the reward is from the perspective of A, B will get the opposite of A.
        if position[1] == 0:
            return 100
        elif position[1] == NUM_COLS - 1:
            return -100
        else:
            return 0

    def get_current_state(self):
        return self.positions, self.ball_owner, self.get_current_index()

    def get_current_index(self):
        player1_index = self.positions[0][0] * NUM_COLS + self.positions[0][1]
        player2_index = self.positions[1][0] * NUM_COLS + self.positions[1][1]
        if player2_index > player1_index:
            player2_index -= 1
        return self.ball_owner * (NUM_ROWS * NUM_COLS * (NUM_ROWS * NUM_COLS - 1)) + player1_index * (NUM_ROWS * NUM_COLS - 1) + player2_index


## Test
# soccer_game = SoccerGame()
# print("Initial State")
# print(soccer_game.get_current_state())
# print(soccer_game.check_target_state())
# print("Move players")
# next_state, reward, done = soccer_game.step([2, 2], 1)
# print("next state is {0}".format(next_state))
# print("reward is {0}".format(reward))
# print("The state is updated to {0}".format(soccer_game.get_current_state()))
# print("The game ends is {0}".format(done))
# print(soccer_game.check_target_state())
