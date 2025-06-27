# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


# Use a raw string to avoid escaping backslashes
model_path = r"C:\Users\brij1\OneDrive\Desktop\PROJECTS\Autisim Project ML\trained_model.sav"
loaded_model = pickle.load(open(model_path, 'rb'))
