import numpy as np
from scipy.spatial.transform import Rotation as R
# /home/qzj/code/catkin_iros23/devel/lib/python3/dist-packages
import os
import sys
sys.path.append("/home/qzj/code/catkin_iros23/devel/lib/python3/dist-packages")
# from openlane_bag.msg import LaneList, Lane, LanePoint
# from geometry_msgs.msg import PoseStamped

def posemsg_to_np(pose_):
    pose = np.eye(4)
    pose[:3, 3] = np.array(pose_[:3])
    pose[:3, :3] = R.from_quat(pose_[3:]).as_matrix()
    return pose

def lanemsg_to_list(lane_list_msg):
    lane_list = []
    for lane_id in range(len(lane_list_msg)):
        lane = lane_list_msg[lane_id]
        if lane['sensor'] == 'cam_0':
            lane_dict = {'xyz': [], 'category': lane['type'], 'visibility': [], 'track_id': lane_id, 'attribute': lane['sensor']}
            if len(lane['points']) > 0:
                for lane_point_id in range(len(lane['points'])):
                    lane_point = lane['points'][lane_point_id]
                    if abs(lane_point['y']) > 100 or abs(lane_point['x']) > 100:
                        continue
                    lane_dict['xyz'].append([lane_point['x'], lane_point['y'], lane_point['z']])
                    lane_dict['visibility'].append(1)
                lane_dict['xyz'] = np.asarray(lane_dict['xyz'])
                lane_dict['visibility'] = np.asarray(lane_dict['visibility'])
                if lane_dict['xyz'] != []:
                    lane_list.append(lane_dict)
    return lane_list