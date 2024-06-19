#!/usr/bin/env python
#!/bin/sh
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
from time import perf_counter
from copy import deepcopy
import tqdm, cv2, json, os, glob
from lane_slam.system.lane_opt import LaneOptimizer
from lane_slam.lane_utils import drop_lane_by_p
from misc.ros_utils.msg_utils import posemsg_to_np
from misc.config import cfg
import numpy as np

class LaneMapping(LaneOptimizer):
    def __init__(self, bag_file, save_result=True):
        super(LaneMapping, self).__init__(bag_file)
        self.load_data()
        self.tracking_init()
        self.graph_init()
        self.debug_init()
        self.save_result = save_result

    def process(self):
        for frame_id, frame_data in enumerate(tqdm.tqdm(self.frames_data, leave=False, dynamic_ncols=True)):
            pose_wc = frame_data['gt_pose']
            timestamp = frame_data['timestamp']
            lane_pts_c = drop_lane_by_p(deepcopy(frame_data['lanes_predict']), p=cfg.preprocess.drop_prob)
            self.time_stamp.append(timestamp)

            # 1. odometry
            t0 = perf_counter()
            self.odometry(lane_pts_c, pose_wc, timestamp)
            self.odo_timer.update(perf_counter() - t0)

            # 2. lane association
            t1 = perf_counter()
            self.lane_association()
            self.assoc_timer.update(perf_counter() - t1)

            # 3. update lanes in map
            self.map_update()
            self.prev_frame = self.cur_frame
            self.whole_timer.update(perf_counter() - t0)
            self.lane_nms(self.cur_frame)

            if self.merge_lane:
                self.post_merge_lane()

            if self.save_result:
                self.save_pred_to_json(lane_pts_c, timestamp)
                self.save_lanes_to_json(self.cur_frame)
                # self.save_for_visualization(self.cur_frame)
        
        self.map_update(last=True)
        stats = self.evaluate_pose()
        stats.update({"map_size": self.map_size()})
        stats.update({"graph": self.graph_build_timer.avg * 1000})
        stats.update({"isam": self.opt_timer.avg * 1000})
        if self.eval_pose_only:
            return stats
        
        self.post_merge_lane()
        # self.post_merge_lane()
        
        self.save_map()
        if self.visualization:
            # stats.update(self.eval_single_segment())
            self.visualize_map()
            self.viz_project()

        return stats

    def lane_nms(self, frame):
        for lane in frame.get_lane_features():
            if lane.id == -1:
                continue
            landmark = self.lanes_in_map[lane.id]
            landmark.add_obs_frame_id(frame.frame_id)

        min_obs_num = 4
        ids_to_remove = []
        for lm in self.lanes_in_map.values():
            if frame.frame_id - lm.obs_first_frame_id < min_obs_num+2:
                continue
            if lm.obs_num < min_obs_num:
                ids_to_remove.append(lm.id)
        for id in ids_to_remove:
            self.lanes_in_map.pop(id)

    def post_merge_lane(self):

        overlap_id = []
        
        for lane_id_a, lane_feature_a in self.lanes_in_map.items():
            for lane_id_b, lane_feature_b in self.lanes_in_map.items():
                if lane_id_a == lane_id_b:
                    continue
                overlap_a = lane_feature_a.overlap_ratio(lane_feature_b)
                overlap_b = lane_feature_b.overlap_ratio(lane_feature_a)
                
                # 线的重叠率阈值
                overlap_ratio = 0.7
                if overlap_a >= overlap_ratio and lane_feature_a.size() < lane_feature_b.size():
                    # l_a, l_b = self.merge_lanes_ab(lane_feature_a, lane_feature_b, theta_threshold=10)
                    # if lane_feature_a == l_a and lane_feature_b == l_b:
                    #     pass
                    # else:
                        # 其中一条线a被另一条线覆盖比例大于overlap_ratio，则将其从地图中移除
                        overlap_id.append(lane_id_a)
                elif overlap_b >= overlap_ratio and lane_feature_b.size() < lane_feature_a.size():
                    # l_a, l_b = self.merge_lanes_ab(lane_feature_a, lane_feature_b, theta_threshold=10)
                    # if lane_feature_a == l_a and lane_feature_b == l_b:
                    #     pass
                    # else:
                        # 其中一条线a被另一条线覆盖比例大于overlap_ratio，则将其从地图中移除
                        overlap_id.append(lane_id_b)
                elif overlap_a * overlap_b > 0:
                    # 两条线有重叠，且夹角较小，则进行合并，theta_threshold是平行四边形角度阈值
                    lane_feature_a, lane_feature_b = self.merge_lanes_ab(lane_feature_a, lane_feature_b, theta_threshold=5)
                else:
                    # 两条线无重叠，且夹角较小，则进行扩展，threshold是距离阈值
                    lane_feature_a, lane_feature_b = self.extend_lane_ab(lane_feature_a, lane_feature_b, threshold=8)

        overlap_id = list(set(overlap_id))
        for lane_id in overlap_id:
            self.lanes_in_map.pop(lane_id)

    def merge_lanes_ab(self, lane_a, lane_b, theta_threshold=5):
        '''
        这个函数功能可以合并两条【有重叠的】、
        【夹角大于一定值的】线，如果两条线类别
        相同，则合并它们，并合并为同一个ID；如
        果两条线类别不同，则合并它们，但两者的
        ID并不发生变化。
        
        例如：
        (1) 同类别
        a: -------------------------
        b:                     ---------
        处理后：
        a: ---------------------
        b:                      --------
        
        (2) 不同类别
        a: -------------------------
        b:                     ++++++++++++
        处理后：
        a: -------------------------
        b:                          +++++++
        
        (3) 有一定夹角（如匝道场景）
        a: ---------------------------------
        b:                                /
                                         /
                                        /
                                       /
        处理后（不做处理）：
        a: ---------------------------------
        b:                                /
                                         /
                                        /
                                       /
        '''
        overlap_points_a = lane_a.get_overlap_points(lane_b)
        overlap_points_b = lane_b.get_overlap_points(lane_a)
        
        if len(overlap_points_a) * len(overlap_points_b) == 0:
            pass
        elif np.all(overlap_points_a[-1] == lane_a.points[-1]) and \
            np.all(overlap_points_b[0] == lane_b.points[0]):
            # self.judge_linking(overlap_points_a[0], overlap_points_a[-1], \
            #     overlap_points_b[0], overlap_points_b[-1], theta_threshold):
            # b线在前，a线在后，并且它们平行
            mid_pnt = (overlap_points_a[-1] + overlap_points_b[0]) / 2
            near_idx_a = np.argmin(np.linalg.norm(lane_a.points-mid_pnt, axis=1))
            near_idx_b = np.argmin(np.linalg.norm(lane_b.points-mid_pnt, axis=1))
            lane_a.points = np.vstack([lane_a.points[:near_idx_a], mid_pnt])
            lane_b.points = np.vstack([mid_pnt, lane_b.points[near_idx_b+1:]])
        elif np.all(overlap_points_b[-1] == lane_b.points[-1]) and \
            np.all(overlap_points_a[0] == lane_a.points[0]):
            # self.judge_linking(overlap_points_a[0], overlap_points_a[-1], \
            #     overlap_points_b[0], overlap_points_b[-1], theta_threshold):
            # a线在前，b线在后，并且它们平行
            mid_pnt = (overlap_points_a[0] + overlap_points_b[-1]) / 2
            near_idx_a = np.argmin(np.linalg.norm(lane_a.points-mid_pnt, axis=1))
            near_idx_b = np.argmin(np.linalg.norm(lane_b.points-mid_pnt, axis=1))
            lane_b.points = np.vstack([lane_b.points[:near_idx_b], mid_pnt])
            lane_a.points = np.vstack([mid_pnt, lane_a.points[near_idx_a+1:]])
        else:
            # 有一定角度，一般是安全岛，不处理
            pass
        
        return lane_a, lane_b
    
    def extend_lane_ab(self, lane_a, lane_b, threshold=5):
        '''
        这个函数可以检测两条线是否可以相连，不论类别是否相同，如
        果两条线断开，且断开的距离【小于一定阈值】，并且它们的
        延长线可以【相交】，则会返回它们的延长线。
        
        例如：
        (1) 平行线
        a: -------------
        b:                     +++++++++
        处理后：
        a: ----------------
        b:                 +++++++++++++
        
        (2) 有一定夹角（如匝道场景）
        a: ---------------------------------
        b:                                /
                                         /
                                        /
                                       /
        处理后（不做处理）：
        a: ---------------------------------
        b:                                /
                                         /
                                        /
                                       /
        '''
        pnt_a = lane_a.points
        pnt_b = lane_b.points
        
        # 平行四边形计算取头或尾多少个点(间距为0.5 m)
        foot_length = 6
        
        if min(np.linalg.norm(pnt_a[0] - pnt_b[-1]), 
               np.linalg.norm(pnt_a[-1] - pnt_b[0])) >= threshold:
            pass
        elif np.linalg.norm(pnt_a[0] - pnt_b[-1]) < threshold:
            # a线在前，b线在后
            if self.judge_linking(pnt_a[0], pnt_a[min(foot_length, len(pnt_a)-1)], \
                pnt_b[-min(foot_length, len(pnt_b)-1)], pnt_b[-1], theta_threshold=5):
                mid_pnt = (pnt_a[0] + pnt_b[-1]) / 2
                add_pnts = self.generate_linear_interpolation_points(mid_pnt, pnt_a[0])
                lane_a.points = np.vstack([add_pnts[:-1], pnt_a])
                add_pnts = self.generate_linear_interpolation_points(pnt_b[-1], mid_pnt)
                lane_b.points = np.vstack([pnt_b, add_pnts[1:]])
        elif np.linalg.norm(pnt_a[-1] - pnt_b[0]) < threshold:
            # b线在前，a线在后
            if self.judge_linking(pnt_b[0], pnt_b[min(foot_length, len(pnt_b)-1)], \
                pnt_a[-min(foot_length, len(pnt_a)-1)], pnt_a[-1], theta_threshold=5):
                mid_pnt = (pnt_b[0] + pnt_a[-1]) / 2
                add_pnts = self.generate_linear_interpolation_points(mid_pnt, pnt_b[0])
                lane_b.points = np.vstack([add_pnts[:-1], pnt_b])
                add_pnts = self.generate_linear_interpolation_points(pnt_a[-1], mid_pnt)
                lane_a.points = np.vstack([pnt_a, add_pnts[1:]])
        
        return lane_a, lane_b
            
    def judge_linking(self, a_st, a_ed, b_st, b_ed, theta_threshold=10):
        '''
        例如：
        | a_ed
        |
        |       | b_ed
        |       |
        | a_st  |
                |
                | b_st
        解析：
        计算平行四边形向量
              v1
         <----------
       -1|^       /|-1
         | \v3   / |
         |  \   /  |
         |   \ /   |
       a |    \    | b
         |   / \   |
         |  /   \  |
         | /v4   \ |
        0|v       \|0
         <----------
              v2
        '''
        
        v_1 = a_ed - b_ed
        v_2 = a_st - b_st
        v_3 = a_ed - b_st
        v_4 = a_st - b_ed
        
        # theta_parallel = self.angle_between_vectors(v_1, v_2)
        theta_vertical = self.angle_between_vectors(v_3, v_4)
        
        if theta_vertical < theta_threshold:
            return True
        else:
            return False
    
    def angle_between_vectors(self,v1,v2):
        dot_product = np.dot(v1,v2)
        cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(cos_angle)*180/np.pi
        
        return theta if theta<90 else 180-theta
    
    def generate_linear_interpolation_points(self, point1, point2, distance=0.5):
        '''
        生成两个3D点之间的线性差值点，间距为给定的距离。
        :param point1: 第一个3D点，格式为(x1, y1, z1)
        :param point2: 第二个3D点，格式为(x2, y2, z2)
        :param distance: 每个差值点之间的固定距离
        :return: 差值点的列表
        '''
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # 计算两点之间的距离
        total_distance = np.linalg.norm(point2 - point1)
        
        # 计算差值点的数量
        num_points = int(np.ceil(total_distance / distance))
        
        # 生成差值点
        points = [point1 + t * (point2 - point1) / num_points for t in range(num_points + 1)]
        
        return np.array(points)
    
    def viz_project(self):
        info = json.load(open('info.json', 'r'))
        imgs_path = np.sort(glob.glob(os.path.join(self.bag_file, 'front_wide/1*.jpg')))
        K = np.array(info['sensors'][0]['intrinsic']['K'])
        D = np.array(info['sensors'][0]['intrinsic']['D'])
        Tc2l = posemsg_to_np(np.array(info['sensors'][0]['extrinsic']['to_lidar_main']))
        Tl2c = np.linalg.inv(Tc2l)
        for frame_id, frame_data in enumerate(tqdm.tqdm(self.frames_data, leave=False, dynamic_ncols=True)):
            pose_wc = frame_data['gt_pose']
            timestamp = str(int(frame_data['timestamp']*1e6))
            img_path = imgs_path[frame_id + 0]
            img = cv2.imread(img_path)
            
            d_T = np.eye(4)
            dist = cv2.undistort(img, K, D)
            for lane in self.lanes_in_map.values():
                category =  lane.category
                xyz =  lane.points
                pnts = np.ones([len(xyz), 4])
                pnts[:, :3] = xyz
                pnts = Tl2c @ (d_T@(np.linalg.inv(pose_wc) @ pnts.T))
                t_in = np.linalg.norm(pnts[:3, :], axis=0) < 100
                pnts = pnts[:, t_in]
                pnts = pnts[:, pnts[2, :] > 0]
                img_pnts = K @ pnts[:3, :]
                img_pnts = (img_pnts/img_pnts[2, :]).T
                clr = (0,0,255) if category == 'lane_solid' else (0,255,0)
                for i, pnt in enumerate(img_pnts[1:]):
                    # cv2.circle(dist, (int(pnt[0]), int(pnt[1])), radius=5, color=clr, thickness=-1)
                    cv2.line(dist, 
                        (int(img_pnts[i, 0]), int(int(img_pnts[i, 1]))), (int(pnt[0]), int(pnt[1])), 
                        color=clr, 
                        thickness = 5)
            dist = cv2.resize(dist, (1600, 900))
            cv2.imshow('1', dist)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()