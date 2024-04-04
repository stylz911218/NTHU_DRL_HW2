import gym
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import numpy as np
import cv2

model_path = '110062342_hw2_data'

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self, model_path):
        self.model = MyDQNModel()
        self.model.set_state_dict(paddle.load(model_path))

    def act(self, observation):
        obs1 = np.expand_dims(observation, axis=0)
        action = self.model(paddle.to_tensor(obs1, dtype='float32'))
        action = paddle.argmax(action).numpy()[0]
        return action

class MyDQNModel(nn.Layer):
    def __init__(self):
        super(MyDQNModel, self).__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)