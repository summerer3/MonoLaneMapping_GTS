cd /root/MonoLaneMapping_offical
rm -rf /root/MonoLaneMapping_offical/self_data
mkdir -p /root/MonoLaneMapping_offical/self_data

ln -s $SENSE_PATH/3D_POLYLINE/lidar /root/MonoLaneMapping_offical/self_data/
ln -s $SLAM_ODOM_PATH /root/MonoLaneMapping_offical/self_data/slam.txt

python /root/MonoLaneMapping_offical/trans_odom.py
python /root/MonoLaneMapping_offical/examples/demo_mapping.py

mkdir -p $SENSE_PATH/global_lane_results/
cp /root/MonoLaneMapping_offical/outputs/lane_mapping/visualization/self_data/map.json $SENSE_PATH/global_lane_results/
