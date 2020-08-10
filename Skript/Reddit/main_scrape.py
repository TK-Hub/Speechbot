#==================================================================================================
#                       Main Script for Speechbot
#                       Author: Tim Kolb
#                       Created: 10.05.2019
#==================================================================================================

import pandas as pd
import os
import json
import numpy as np


path_to_json = 'C:/Users/735477/Desktop/Trumpbot/Data'
json_files = os.listdir(path_to_json)
files = [os.path.join(path_to_json, i) for i in json_files]
print(files)

data = []
for vidfile in files:
    with open(vidfile, "r", encoding="utf8") as read_file:
        newdata = json.load(read_file)
        data = data + newdata

print(len(data))

convos = []
for i in data:
    if i["hasReplies"] == True:
        convos.append(i)

print(len(convos))

with open('C:/Users/735477/Desktop/Trumpbot/Full_Data.json', 'w') as fout:
    json.dump(convos, fout)

#print(data[0])

sumo = 0
for i in convos:
    print(i["numberOfReplies"])
    sumo = sumo + i["numberOfReplies"]

print(sumo)
