# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:32:28 2021

@author: Abhiram
"""

import glassdoor_scraping as gs
import pandas as pd
path = "C:/Users/Abhiram/Desktop/Major Projects/Data Scientist Salary Prediction/chromedriver.exe"

df = gs.get_jobs('data scientist',1000, False, path, 15)
df.to_csv('glassdoor_jobs.csv', index = False)
