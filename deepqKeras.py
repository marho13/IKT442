# -*- coding: utf-8 -*-
import cv2
import gym
import time
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import win32gui, win32ui, win32con, win32api
from statistics import mode, mean
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check

WIDTH = 1920
HEIGHT = 1080
how_far_remove = 800
log_len = 25
rs = (20, 15)

motion_req = 800
motion_log = deque(maxlen=log_len)

choices = deque([], maxlen=6)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)


class motion:
    def __init__(self):
        pass

    def delta_images(self, t0, t1, t2):
        d1 = cv2.absdiff(t2, t0)
        return d1

    def motion_detection(self, t_minus, t_now, t_plus):
        delta_view = self.delta_images(t_minus, t_now, t_plus)
        retval, delta_view = cv2.threshold(delta_view, 16, 255, 3)
        cv2.normalize(delta_view, delta_view, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(delta_view, cv2.COLOR_RGB2GRAY)
        delta_count = cv2.countNonZero(img_count_view)
        dst = cv2.addWeighted(t_now, 1.0, delta_view, 0.6, 0)
        delta_count_last = delta_count
        return delta_count


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def actionChoice(action):
    choice_picked = ''
    if action == 0:
        straight()
        choice_picked = 'straight'

    elif action == 1:
        reverse()
        choice_picked = 'reverse'

    elif action == 2:
        forward_left()
        choice_picked = 'forward+left'

    elif action == 3:
        forward_right()
        choice_picked = 'forward+right'

    elif action == 4:
        reverse_left()
        choice_picked = 'reverse+left'

    elif action == 5:
        reverse_right()
        choice_picked = 'reverse+right'

    elif action == 6:
        no_keys()
        choice_picked = 'nokeys'

    return choice_picked


if __name__ == "__main__":
    Game = "Udacity-car-simulation"
    state_size = 1920 * 1080 * 3
    batch_size = 32
    action_size = 6
    mode_choice = 0
    paused = False
    done = False

    screen = grab_screen(region=(0, 40, WIDTH, HEIGHT + 40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH, HEIGHT))

    t_minus = prev
    t_now = prev
    t_plus = prev

    agent = DQNAgent(state_size, action_size)

    m = motion()

    try:
        agent.load("./save/{}.h5".format(Game))

    except:
        print("Was unable to load model")

    while True:

        if not paused:

            # Check if outside track
            # If outside, pause and make done true

            last_time = time.time()

            # Screen stuff
            screen = grab_screen(region=(0, 40, WIDTH, HEIGHT + 40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            delta_count = m.motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen

            # End of screen stuff

            # Starting
            prediction = agent.act(screen)

            action = np.argmax(prediction)
            choice_picked = actionChoice(action)
            motion_log.append(delta_count)
            motion_avg = round(mean(motion_log), 3)
            print('loop took {} seconds. Motion: {}. Choice: {}'.format(round(time.time() - last_time, 3), motion_avg,
                                                                        choice_picked))

            reward = 10
            next_state = grab_screen(region=(0, 40, WIDTH, HEIGHT + 40))
            next_state = cv2.resize(next_state, (WIDTH, HEIGHT))
            agent.remember(screen, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

