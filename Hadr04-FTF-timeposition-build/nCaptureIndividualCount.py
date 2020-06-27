#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:31:35 2020

@author: carlopalazzi
"""

import pandas as pd

df = pd.read_csv('timepositions.csv', names=['t', 'x', 'y', 'z'])

count = len(df)

with open('nCaptureIndividualCount.csv','a') as fd:
    fd.write(str(count)+'\n')