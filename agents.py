import numpy as np
from grid2op.Agent import MLAgent
from grid2op.Converter import ToVect

import tensorflow as tf
from tensorflow import keras


class my_test_agent(MLAgent):

    def __init__(self, ENV, action_space_converter=ToVect, **kwargs_converter):
        MLAgent.__init__(self, ENV.action_space, action_space_converter, **kwargs_converter)
        self.action_space = ENV.action_space
        self.do_nothing_act = self.action_space()
        self.converter = ToVect(self.action_space)  # (64, 200, 200, 3)
        self.converter.seed(0)
        self.converter.init_converter(max_sub_changed=ENV.parameters.MAX_SUB_CHANGED)

        print(self.do_nothing_act.to_vect())

        ##BUilding the modeL:

        # Configuration paramaters for the whole setup
        seed = 42
        gamma = 0.99  # Discount factor for past rewards
        epsilon = 1.0  # Epsilon greedy parameter
        epsilon_min = 0.1  # Minimum epsilon greedy parameter
        epsilon_max = 1.0  # Maximum epsilon greedy parameter
        epsilon_interval = (
                epsilon_max - epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        batch_size = 32  # Size of batch taken from replay buffer
        max_steps_per_episode = 10000

        sample_obs_vect = self.converter.convert_obs(ENV.reset())
        sample_obs_vect.reshape(-1, len(sample_obs_vect), 1)
        print(len(sample_obs_vect))
        inputs = keras.Input(shape=(len(sample_obs_vect),1))
        # Apply some convolution and pooling layers
        x = keras.layers.Conv1D(filters=32, kernel_size=(3), activation="relu")(inputs)
        x = keras.layers.MaxPooling1D(pool_size=(3))(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=(3), activation="relu")(x)
        x = keras.layers.MaxPooling1D(pool_size=(3))(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=(3), activation="relu")(x)

        # Apply global average pooling to get flat feature vectors
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Add a dense classifier on top
        num_classes = len(self.do_nothing_act.to_vect())
        print(num_classes)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.my_neural_network = model
        # self.my_neural_networl.load(path)

    def convert_obs(self, observation):
        # convert the observation
        return np.concatenate((observation.load_p, observation.rho + observation.p_or))

    def convert_act(self, encoded_act):
        return self.converter.convert_act(encoded_act)

    def act(self, obs, reward, done=False):
        converted_obs = self.convert_obs(obs)
        self.model.predict
        return self.do_nothing_act
        #return self.convert_from_vect(self.do_nothing_vect)

    def my_act(self, transformed_observation, reward, done=False, learning=False):
        return self.do_nothing_act
