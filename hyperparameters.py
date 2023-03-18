'''
file: dependencies.py
author: @Vincit0re
objective: Set all the hyperparameters for the project. This file contains Hyperparameters class that have all the hyperparameters that are used in the project.
date: 2023-03-13
'''

from dependencies import *

# define a hyperparameter class and set a global variable color_maps as dictionary to it


class Hyperparameters:
    '''This is a class for setting all the hyperparameters for the project. This class contains all the hyperparameters that are used in the project.'''
    # global variable color_maps as dictionary
    _COLOR_MAPS = {
        'simple': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
        'rainbow': [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)],
        'ocean': [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0)],
        'summer': [(0, 255, 255), (0, 255, 127), (0, 255, 0), (127, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0)],
        'spring': [(255, 0, 255), (255, 127, 255), (255, 255, 255), (255, 255, 127), (255, 255, 0), (255, 127, 0), (255, 0, 0)],
        'autumn': [(255, 0, 0), (255, 127, 0), (255, 255, 0), (255, 255, 127), (255, 255, 255), (127, 255, 255), (0, 255, 255)],
        'winter': [(0, 0, 255), (127, 0, 255), (255, 0, 255), (255, 0, 127), (255, 0, 0), (255, 127, 0), (255, 255, 0)],
        'cool': [(0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0)]
    }
    _thresh = 180
    _thresh_type = 'TOZERO_INV'
    _color_map = 'cool'
    _show = True
    _save = True
    _save_path = 'output'
