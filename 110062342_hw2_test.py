import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import numpy as np
import gym
from gym.spaces import Box
from PIL import Image

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

    def load_model(self, model_path):
        model = Model(num_inputs=4, num_actions=12)  # 修改为你的输入维度和动作空间维度
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        return model


    def act(self, observation):
        observation = preprocess_observation(observation)
        observation = np.expand_dims(observation, axis=0)
        observation = paddle.to_tensor(observation, dtype='float32')
        action = self.model(observation)
        action = np.squeeze(paddle.argmax(action).numpy())
        return action

def preprocess_observation(observation, target_size=(4, 84, 84)):
    observation = np.array(observation, dtype=np.uint8)  # 將數據類型轉換為8位元整數
    # 將ndarray轉換為PIL Image
    image = Image.fromarray(observation)

    # 將圖像轉換為灰度
    image = image.convert('L')

    # Resize圖像
    image = image.resize((target_size[1], target_size[2]))

    # 將PIL Image轉換回ndarray
    processed_observation = np.array(image)

    # 將通道維度移到第一個維度
    processed_observation = np.expand_dims(processed_observation, axis=0)

    # 將圖像復制多次以構建 (4, 84, 84) 的形狀
    processed_observation = np.tile(processed_observation, (target_size[0], 1, 1))

    return processed_observation
