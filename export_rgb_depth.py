from pathlib import Path
from rosbags.highlevel import AnyReader
import cv2
import numpy as np
import os

bagname = 'Group_11_object_02.bag'

rgb_dir = 'object02_rgb'
depth_dir = 'object02_depth'

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

with AnyReader([Path(bagname)]) as reader:
    connections = [x for x in reader.connections if x.topic in [
        '/camera/color/image_raw',
        '/camera/aligned_depth_to_color/image_raw'
    ]]

    rgb_idx = 0
    depth_idx = 0

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        if connection.topic == '/camera/color/image_raw':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if rgb_idx % 10 == 0:
                cv2.imwrite(f'{rgb_dir}/{rgb_idx:04d}.png', img)
            rgb_idx += 1

        elif connection.topic == '/camera/aligned_depth_to_color/image_raw':
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            if depth_idx % 10 == 0:
                cv2.imwrite(f'{depth_dir}/{depth_idx:04d}.png', depth)
            depth_idx += 1

print('Done')