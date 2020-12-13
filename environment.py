import collections
import cv2
import numpy as np
import gym


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, env_name):
        super(PreprocessFrame, self).__init__(env)

        h, w, c = env.observation_space.shape
        

        # Crop and scale per game
        if 'Breakout' in env_name:
            self.crop = (32, 0, 8)
            scale = 0.475
        if 'SpaceInvaders' in env_name:
            self.crop = (0, 0, 0)  # top bottom left/right
            scale = 0.425
        if 'Krull' in env_name:
            self.crop = [10, 34, 0]
            scale = 0.475
        if 'Pong' in env_name:
            self.crop = [33, 14, 0]
            scale = 0.475
        if 'StarGunner' in env_name:
            self.crop = [30, 20, 0]
            scale = 0.485
        

        h = h - self.crop[0] - self.crop[1]
        w -= self.crop[2] * 2
        # Ignoring color channel, completely arbitrary.
        # print("Calc cropped:", h, w)

        h = int(h * scale)
        w = int(w * scale)

        self.shape = (h, w)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,  # Storing them in original 0-255 b/c much MUCH more memory efficient.
        )
    

    def observation(self, obs):
        cropped = obs[self.crop[0]:-(self.crop[1]+1), self.crop[2]:-(self.crop[2]+1)]
        # print("Actual cropped", cropped.shape)
        # Convert color from RGB to grayscale!
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
            # cv2.resize wants (width, height)
        # print("Actual resized", resized.shape)

        return resized


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat=4):
        super(StackFrames, self).__init__(env)
        
        self.shape = (repeat, *env.observation_space.shape)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8
        )

        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return [observation for observation in self.stack]
    

    def observation(self, observation):
        self.stack.append(observation)

        return [observation for observation in self.stack]
        


def make_env(env_name):
    env = gym.make(env_name)
    
    env = PreprocessFrame(env, env_name)
    env = StackFrames(env)

    return env
