
from collections import namedtuple
import pygame
from pygame.locals import *
import logging


settings = {
    # Display
    'width': 1000,
    'height': 600,
    'gui': False,
    'fps': 15,

    # Main loop
    'update_interval': 0.01, # 0.33,
    'turn_based': False,
    'n_rounds': 1500,
    'save_replay': False,
    'make_video_from_replay': False,

    # Game properties
    'cols': 17,
    'rows': 17,
    'grid_size': 30,
    'crate_density': 0.75,#0.75,
    'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'],
    'max_agents': 4,
    'max_steps': 400,  #400
    'stop_if_not_training': True,
    'bomb_power': 3,
    'bomb_timer': 4,
    'explosion_timer': 2,

    # Rules for agents
    'timeout': 5.0,
    'reward_kill': 5,
    'reward_coin': 1,
    'reward_slow': -1,

    # User input
    'input_map': {
        K_UP: 'UP',
        K_DOWN: 'DOWN',
        K_LEFT: 'LEFT',
        K_RIGHT: 'RIGHT',
        K_RETURN: 'WAIT',
        K_SPACE: 'BOMB',
    },

    # Logging levels
    'log_game': logging.INFO,
    'log_agent_wrapper': logging.INFO,
    'log_agent_code': logging.DEBUG,
}
settings['grid_offset'] = [(settings['height'] - settings['rows']*settings['grid_size'])//2] * 2
s = namedtuple("Settings", settings.keys())(*settings.values())


events = [
    'MOVED_LEFT',#0
    'MOVED_RIGHT',#1
    'MOVED_UP',#2
    'MOVED_DOWN',#3
    'WAITED',#4
    'INTERRUPTED',#5
    'INVALID_ACTION',#6

    'BOMB_DROPPED',#7
    'BOMB_EXPLODED',#8

    'CRATE_DESTROYED',#9
    'COIN_FOUND',#10
    'COIN_COLLECTED',#11

    'KILLED_OPPONENT',#12
    'KILLED_SELF',#13

    'GOT_KILLED',#14
    'OPPONENT_ELIMINATED',#15
    'SURVIVED_ROUND',#16
]
e = namedtuple('Events', events)(*range(len(events)))
