import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import numpy as np
import cv2
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
        self.framestack = StackFrame(num_stack=4)
        self.action = None
        self.ep = 0

    def load_model(self, model_path):
        model = Model(num_inputs=4, num_actions=12)
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        return model


    def act(self, observation):
        observation = preprocess_observation(observation)
        if self.current_frame == self.skip or self.action == None or self.ep == 1623:
            if self.action == None or self.ep == 1623:
                obs = self.framestack.reset(observation=observation)
                self.ep = 0
            else:
                obs = self.framestack.update(observation=observation)
            obs1 = np.expand_dims(obs, axis=0)
            obs2 = paddle.to_tensor(obs1, dtype='float32')
            action = self.model(obs2)
            action = np.squeeze(paddle.argmax(action).numpy())
            action = action.item()
            self.action = action
            self.current_frame = 1
            self.ep += 1
            return action
        else:
            self.current_frame += 1
            self.ep += 1
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

    return togray_observation

class StackFrame:

    # Initialize
    def __init__(self, num_stack) -> None:
        self.frames = deque(maxlen=num_stack)
        self.num_stack = num_stack

    # Reset stack -> fill up
    def reset(self, observation):
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.frames
    
    # Step
    def update(self, observation):
        self.frames.append(observation)
        return self.frames
