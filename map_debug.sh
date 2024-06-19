rm self_data
ln -s ../data/rain05 self_data
python trans_odom.py
python examples/demo_mapping.py