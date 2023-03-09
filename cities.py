# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:36:08 2023

@author: talam
"""
import folium
import json


f=open('./data/5m-US-counties.json')

geo=json.load(f)