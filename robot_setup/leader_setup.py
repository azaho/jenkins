# robot_setup/leader_setup.py

import numpy as np
from ikpy.link import OriginLink, URDFLink
from ikpy.chain import Chain
from dynamixel_sdk import PortHandler, PacketHandler

# ----------------------------
# Define Leader Kinematic Chain
# ----------------------------
joint1 = URDFLink(
    name="1_base_rotation",
    origin_translation=np.array([0, 0, 0.080]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 0, 1]),
    joint_type="revolute",
)

joint2 = URDFLink(
    name="2_shoulder_tilt",
    origin_translation=np.array([0, 0, 0.018]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 1, 0]),
    joint_type="revolute",
)

L2 = 0.095
joint3 = URDFLink(
    name="3_elbow_tilt",
    origin_translation=np.array([0, 0, L2]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 1, 0]),
    joint_type="revolute",
)

L3 = 0.100
joint4 = URDFLink(
    name="4_wrist_tilt",
    origin_translation=np.array([L3, 0, 0]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, -1, 0]),
    joint_type="revolute",
)

L4 = 0.120
joint5 = URDFLink(
    name="5_gripper_rotate",
    origin_translation=np.array([0, 0, -L4]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([1, 0, 0]),
    joint_type="revolute"
)

robot_chain = Chain(name="robot_chain", links=[
    OriginLink(),
    joint1,
    joint2,
    joint3,
    joint4,
    joint5
])

# ----------------------------
# Dynamixel Setup for Leader
# ----------------------------
PROTOCOL_VERSION = 2.0
DXL_IDS = [1, 2, 3, 4, 5, 6]
BAUDRATE = 1e6
DEVICENAME = '/dev/tty.usbmodem58CD1775601'
TICKS_PER_REV = 4096


def rad_to_dxl(angle_rad):
    return int((angle_rad / (2 * np.pi)) * TICKS_PER_REV)


def dxl_to_rad(ticks):
    return (ticks / TICKS_PER_REV) * 2 * np.pi


port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)
if not port_handler.openPort():
    raise RuntimeError("Failed to open port for leader!")
if not port_handler.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baudrate for leader!")


def to_signed(val):
    if val > 0x7FFFFFFF:
        return val - 0x100000000
    return val


def read_current_angles():
    angles = []
    for dxl_id in DXL_IDS:
        pos, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, dxl_id, 132)
        if comm_result != 0 or error != 0:
            print(f"Error reading servo {dxl_id}, assuming 0.")
            angles.append(0)
        else:
            angle = dxl_to_rad(to_signed(pos))
            angles.append(angle)
    return np.array(angles)


def forward_kinematics(angles):
    truncated = angles[:5]
    full_angles = np.concatenate(([0], truncated))
    return robot_chain.forward_kinematics(full_angles)


if __name__ == "__main__":
    try:
        current_angles = read_current_angles()
        print("Current joint angles:", current_angles)
        print("\nRobot kinematic chain setup for leader arm is successful!")
    except Exception as e:
        print(f"Error during robot setup: {str(e)}")
