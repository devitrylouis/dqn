def train_explore(agent, env, epoch, prefix=''):
    # Number of won games
    score, loss = 0, 0

    for e in range(epoch):
        # Set a decaying epsilon
        agent.set_epsilon(0.6 - (0.6 - 0.1)*e/epoch)

        # Reset the environment for each epoch
        current_state = env.reset()

        # Initialize metrics
        win, lose = 0, 0

        # Keep playing the game as long as possible
        game_over = False
        while not game_over:

            # The agent follows his policy
            action = agent.act(current_state)

            # Update of the environment in consequence
            next_state, reward, game_over = env.act(action, train = True)

            # Update the metrics
            if reward > 0:
                win += reward
            else:
                lose -=  reward

            # Execute the reinforcement learning strategy
            loss = agent.reinforce(current_state, next_state, action, reward, game_over)

        # Save as a mp4
        if e % 10 == 0:
            env.draw(prefix+str(e))

        # Update stats
        score += win-lose

        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win/lose count {}/{} ({})"
              .format(e, epoch, loss, win, lose, win-lose))
        agent.save(name_weights = prefix+'model.h5', name_model = prefix + 'model.json')

class EnvironmentExploring(Environment):
    super(EnvironmentExploring, self)
    def __init__(self):
        super(EnvironmentExploring, self).__init__()
        # The malus indicators is the only difference
        self.malus_position=np.zeros((self.grid_size, self.grid_size))

    def act(self, action, train=False):

        self.get_frame(int(self.time))

        self.position = np.zeros((self.grid_size, self.grid_size))

        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1

        self.position[self.x, self.y] = 1
        if action == 0:
            if self.x == self.grid_size-3:
                self.x = self.x-1
            else:
                self.x = self.x + 1
        elif action == 1:
            if self.x == 2:
                self.x = self.x+1
            else:
                self.x = self.x-1
        elif action == 2:
            if self.y == self.grid_size - 3:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3:
            if self.y == 2:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        self.time = self.time + 1

        # Set the to-be cumulative reward to zero
        reward = 0

        # Penalize the agent if train is set to True
        if train: reward -= self.malus_position[self.x, self.y]
        self.malus_position[self.x, self.y] = 0.1

        # Add true reward and remove it from the board
        reward += self.board[self.x, self.y]
        self.board[self.x, self.y] = 0

        # Return new state
        game_over = self.time > self.max_time
        state = np.concatenate((self.malus_position.reshape(self.grid_size, self.grid_size, 1),
                                self.board.reshape(self.grid_size, self.grid_size, 1),
                                self.position.reshape(self.grid_size, self.grid_size, 1))
                                ,axis=2)
        state = state[self.x-2:self.x+3,self.y-2:self.y+3,:]

        return(state, reward, game_over)

    def reset(self):
        """This function resets the game and returns the initial state"""

        self.x = np.random.randint(3, self.grid_size-3, size=1)[0]
        self.y = np.random.randint(3, self.grid_size-3, size=1)[0]

        bonus = 0.5*np.random.binomial(1, self.temperature,size=self.grid_size**2)
        bonus = bonus.reshape(self.grid_size,self.grid_size)

        malus = -1.0*np.random.binomial(1, self.temperature,size=self.grid_size**2)
        malus = malus.reshape(self.grid_size, self.grid_size)

        self.to_draw = np.zeros((self.max_time+2, self.grid_size*self.scale, self.grid_size*self.scale, 3))

        malus[bonus>0]=0

        self.board = bonus + malus

        # This makes the dimension of the problem work
        self.malus_position = np.zeros((self.grid_size, self.grid_size))
        self.position = np.zeros((self.grid_size, self.grid_size))
        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1
        self.board[self.x,self.y] = 0
        self.time = 0

        state = np.concatenate((self.malus_position.reshape(self.grid_size, self.grid_size,1),
                                self.board.reshape(self.grid_size, self.grid_size,1),
                                self.position.reshape(self.grid_size, self.grid_size,1)),axis=2)

        state = state[self.x - 2:self.x + 3, self.y - 2:self.y + 3, :]
        return(state)
