"""
This program merge the 3d lanes from single frame
3d lane results.
Copyright: liushilei@baidu.com (>_<)
"""

import json, glob, os
import numpy as np
from tqdm import tqdm

def load_config(config_path):
    """load the config file 

    Args:
        config_path (string): the path of config file 

    Returns:
        dict: the dict of configs
    """
    data = []
    if config_path.split('.')[-1] == 'json':
        data = json.load(open(config_path, 'r'))
    elif config_path.split('.')[-1] == 'txt':
        lines = open(config_path, 'r').readlines()
        for ln in lines:
            ln = ln.strip().split(' ')
            timestamp = ln[0]
            t_xyz_q_xyzw = np.asarray(ln[1:], dtype='float')
            data.append({
                'timestamp': float(timestamp),
                'T_Q': t_xyz_q_xyzw
            })
    
    return data

def get_T_Q(timestamp, data):
    """get the extrinsic [tx, ty, tz, qx, qy, qz, qw] 
    and the weight of current QT

    Args:
        timestamp (float): the time of key frame of SLAM
        data (dict): the dict of single frame

    Returns:
        T_Q_st, T_Q_ed, fraction: the extrinsic [tx, ty, tz, qx, qy, qz, qw] 
    and the weight of current QT
    """
    i = 0
    for i in range(len(data) - 1):
        if data[i]['timestamp'] < timestamp:
            i += 1
        else:
            break
        
    if i > 0:
        T_Q_st = data[i - 1]
        T_Q_ed = data[i]
        fraction = (timestamp - data[i - 1]['timestamp']) / ((data[i]['timestamp'] - data[i - 1]['timestamp']))
    elif i == 0:
        T_Q_st = data[0]
        T_Q_ed = data[0]
        fraction = 0
    
    return T_Q_st, T_Q_ed, fraction

if __name__ == '__main__':
    """
    run this src, will merge single frame to global merged 3d lanes
    """
    json_path = 'self_data/odom_lidar_reference.txt'
    data = load_config(json_path)
    json_paths = np.sort(glob.glob('self_data/lidar/*.json'))
    
    pose_new = 'self_data/pose_new.txt'
    f = open(pose_new, 'w')
    pose_new_txt = ''
    for iidx, json_pth in tqdm(enumerate(json_paths), total=len(json_paths)):
        tm_str = os.path.basename(json_pth).split('.')[0]
        timestamp = np.asarray(tm_str, dtype='float')
        T_Q_last, T_Q_end, fraction = get_T_Q(timestamp, data)
        T_Q = T_Q_end['T_Q'] * fraction + T_Q_last['T_Q'] * (1 - fraction)
        pose_new_txt += str(int(timestamp)) + ' '
        for t in T_Q:
            pose_new_txt += str(t) + ' '
        pose_new_txt += '\n'
    f.writelines(pose_new_txt)
    f.close()
    