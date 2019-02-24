import copy
class Double_DQN(Agent):
    def __init__(self, grid_size, lr = 0.1, discount=0.9, epsilon=0.1, memory_size=100, batch_size=16, n_state=3):
        super(Double_DQN, self).__init__(epsilon = epsilon)

        # Discount and grid size
        self.discount = discount
        self.grid_size = grid_size

        # Number of state
        self.n_state = n_state

        # Memory
        self.memory = Memory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

        # Learning rate
        self.lr = lr

    def learned_act(self, s):
        prediction = self.model.predict(np.array([s,]))
        return(np.argmax(prediction))

    def reinforce(self, s_, n_s_, a_, r_, game_over_):
        # Two steps: first memorize the states, second learn from the pool
        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        # Initializations
        input_states = np.zeros((self.batch_size, 5, 5, self.n_state))
        target_q = np.zeros((self.batch_size, 4))

        # Indicator variable for which model should be trained
        zero_one = np.random.randint(2)

        for i in range(self.batch_size):
            # Random Access Memories
            [s_batch, n_s_batch, a_batch, r_batch, game_over_batch] = self.memory.random_access()
            input_states[i] = s_batch

            # Prediction depending of the model
            if zero_one == 0:
                target_q[i] = self.model.predict(np.array([s_batch]))
            else :
                target_q[i] = self.second_model.predict(np.array([s_batch]))

            if game_over_:
                target_q[i, a_batch] = r_batch
            else:
                # Prediction depending of the model
                if zero_one == 0:
                    prediction1 = self.model.predict(np.array([n_s_batch]))
                    prediction2 = self.second_model.predict(np.array([n_s_batch]))
                else:
                    prediction2 = self.model.predict(np.array([n_s_batch]))
                    prediction1 = self.second_model.predict(np.array([n_s_batch]))

                target_q[i, a_batch] = r_batch + self.discount * (prediction2.ravel())[np.argmax(prediction1)]

        # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
        target_q = np.clip(target_q, -3, 3)

        # Training depending on the model
        if zero_one == 0:
            l = self.model.train_on_batch(input_states, target_q)
        else:
            l = self.second_model.train_on_batch(input_states, target_q)

        return l

    def save(self, name_weights='model.h5', name_model='model.json'):
        self.model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def load(self, name_weights='model.h5', name_model='model.json'):
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("sgd", "mse")
        self.model = model

class Double_DQN_CNN(Double_DQN):
    def __init__(self, *args,**kwargs):
        super(Double_DQN_CNN, self).__init__(*args,**kwargs)

        model = Sequential()

        model.add(Conv2D(32, (1,1), strides=(1,1), input_shape=(5, 5, self.n_state), data_format = 'channels_last'))
        model.add(Conv2D(16, (1,1), strides=(1,1), data_format = 'channels_last'))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('relu'))

        model.compile(sgd(lr = 0.1, decay = 1e-4, momentum = 0.8), "mse")
        self.model = model

        self.second_model = copy.deepcopy(self.model)
        self.second_model.compile(sgd(lr = 0.1, decay = 1e-4, momentum = 0.8), "mse")
        self.second_model.set_weights(self.model.get_weights())
