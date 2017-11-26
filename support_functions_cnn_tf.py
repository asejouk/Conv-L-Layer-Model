#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:21:01 2017

@author: asejouk
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import tensorflow as tf
from PIL import Image
from scipy import ndimage


def create_placeholder(n_H,n_W,n_C,n_Y):
    
    