#!/usr/bin/env python

"""
under development.
"""

import numpy as np
from nuscenes.nuscenes import NuScenes

channel = 5
dataroot = '/home/kosuke/dataset/nuScenes/'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=dataroot, verbose=True)

my_scene = nusc.scene[0]
token = my_scene['first_sample_token']
print(token)
my_sample = nusc.get('sample', token)
lidar = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

pc_flatten = np.fromfile(dataroot + lidar['filename'], dtype=np.float32)
pc = pc_flatten.reshape(-1, channel)
print(pc.shape)

_, boxes, _ = nusc.get_sample_data(lidar['token'], box_vis_level=0)
print(len(boxes))
print(boxes[0])
