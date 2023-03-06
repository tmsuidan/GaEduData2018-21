# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:31:37 2023

@author: talam
"""
import json
import pandas as pd


ga_json='./data/5m-US-counties.json'
with open(ga_json,'r') as f:
    data=json.load(f)