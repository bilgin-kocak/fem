# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:01:57 2022

@author: kocak
"""

import numpy as np

def read_line(line, name):
    if ";" in line:
        line_list = line.replace(";","").strip().split()
        if name in ['LINES', 'QUADS', 'PNT']:
            line_list = list(map(int, line_list))
        else:
            line_list = list(map(float, line_list))
    else:
        line_list = line.strip().split()
        if name in ['LINES', 'QUADS', 'PNT']:
            line_list = list(map(int, line_list))
        else:
            line_list = list(map(float, line_list))
    return line_list

def read_mesh(file_name):
    f = open(f"./meshes/{file_name}", "r")
    # print(f.read())
    file_str = f.read()
    
    file_lines = file_str.split('\n')
        
    msh  = {}
    
    temp_data = []
    write = False
    for line in file_lines:
        if "POS" in line:
            name = "POS"
            write = True
            continue
        elif "LINES" in line:
            name = "LINES"
            write = True
            continue
        elif "QUADS" in line:
            name = "QUADS"
            write = True
            continue
        elif "PNT" in line:
            name = "PNT"
            write = True
            continue
        
        if "];" in line:
            msh[name] = np.array(temp_data)
            temp_data = []
            write = False
            continue
        if write:
            temp_data.append(read_line(line, name))
            
            
            
    return msh
    
    

