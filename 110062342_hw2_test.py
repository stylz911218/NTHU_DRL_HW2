import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import numpy as np
import gym
from gym.spaces import Box
from PIL import Image
from collections import deque

class Model(nn.Layer):

    # Init
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2D(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2D(64, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3136, 512)
        self.fc = nn.Linear(512, num_actions)

    # Forward
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        return self.fc(x)

class Agent(object):
    def __init__(self):
        model_path = '110062342_hw2_data.py'
        self.model = self.load_model(model_path)
        self.skip = 4
        self.current_frame = 1
        self.action = None

    def load_model(self, model_path):
        model = Model(num_inputs=4, num_actions=12)
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        return model


    def act(self, observation):
        if self.current_frame == self.skip or self.action == None:
            observation = preprocess_observation(observation)
            obs = np.expand_dims(observation, axis=0)
            obs1 = paddle.to_tensor(obs, dtype='float32')
            action = self.model(obs1)
            action = paddle.argmax(action).numpy()[0]
            self.action = action
            self.current_frame = 1
            return action
        else:
            self.current_frame += 1
            return self.action


def preprocess_observation(observation):
    
    # Resize
    transforms = T.Compose(
        [T.Resize((84,84)), T.Normalize(0, 255, data_format='HWC')]
    )
    resized_observation = transforms(observation)
    
    # To Gray
    transfrom = T.Grayscale()
    togray_observation = transfrom(resized_observation)
    togray_observation = np.transpose(togray_observation, (2, 0, 1)).squeeze(0)
    
    # Stack observation
    stacked_observation = np.stack([togray_observation] * 4, axis=0)

    return stacked_observation
