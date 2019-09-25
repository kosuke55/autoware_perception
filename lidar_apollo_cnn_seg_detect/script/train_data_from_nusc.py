#!/usr/bin/env python

"""
under development.
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, List


def view_points(points: np.ndarray,
                view: np.ndarray,
                normalize: bool) -> np.ndarray:

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def get_color(category_name: str) -> Tuple[int, int, int]:
    """ Provides the default colors based on the category names. """
    if category_name in ['vehicle.bicycle', 'vehicle.motorcycle']:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name:
        return 255, 158, 0  # Orange
    elif 'human.pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


def points_in_box2d(box2d: np.ndarray, points: np.ndarray):
    p1 = box2d[0]
    p_x = box2d[1]
    p_y = box2d[3]

    i = p_x - p1
    j = p_y - p1
    v = points - p1

    iv = np.dot(i, v)
    jv = np.dot(j, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask = np.logical_and(mask_x, mask_y)

    return mask


grid_range = 60
size = 640
# size = 10
gsize = 2 * grid_range / size

channel = 5
dataroot = '/home/kosuke/dataset/nuScenes/'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=dataroot, verbose=True)

my_scene = nusc.scene[0]
token = my_scene['first_sample_token']

ref_chan = 'LIDAR_TOP'

my_sample = nusc.get('sample', token)
my_sample = nusc.sample[20]

sd_record = nusc.get('sample_data', my_sample['data'][ref_chan])
sample_rec = nusc.get('sample', sd_record['sample_token'])
chan = sd_record['channel']

pc, times = LidarPointCloud.from_file_multisweep(
    nusc, sample_rec, chan, ref_chan, nsweeps=10)
_, boxes, _ = nusc.get_sample_data(sd_record['token'], box_vis_level=0)

# not needed. This is equal to points = pc.points[:3, :]
points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))

axes_limit = grid_range
colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(0, 0, 'x', color='black')
ax.grid(which="major", color="silver")
ticks = np.arange(-grid_range, grid_range + gsize, gsize)

grid_x, grid_y = np.meshgrid(ticks, ticks)
grid_x = grid_x.flatten()
grid_y = grid_y.flatten()


plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlim(-grid_range, grid_range)
ax.set_ylim(-grid_range, grid_range)
for box in boxes:
    c = np.array(get_color(box.name)) / 255.0
    box.render(ax, view=np.eye(4), colors=(c, c, c))

box = boxes[2]
view=np.eye(4)
corners = view_points(box.corners(), view, normalize=False)[:2, :]
box2d = corners.T[[2, 3, 7, 6]]
plt.scatter(box2d[:, 0], box2d[:, 1], marker='^', s=100)

for i in range(len(grid_x)):
    grid_center = np.array([grid_x[i] + gsize / 2, grid_y[i] + gsize / 2])
    print(i)
    fill_area = np.array([[(grid_center[0] - gsize / 2),
                           (grid_center[0] + gsize / 2),
                           (grid_center[0] + gsize / 2),
                           (grid_center[0] - gsize / 2)],
                          [(grid_center[1] + gsize / 2),
                           (grid_center[1] + gsize / 2),
                           (grid_center[1] - gsize / 2),
                           (grid_center[1] - gsize / 2)]])
    if(points_in_box2d(box2d, grid_center)):
        plt.fill(fill_area[0], fill_area[1], color="r", alpha=0.5)
plt.show()

