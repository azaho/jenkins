# robot_setup/follower_setup.py

import numpy as np
from ikpy.link import OriginLink, URDFLink
from ikpy.chain import Chain
from dynamixel_sdk import PortHandler, PacketHandler

# ----------------------------
# Define Follower Kinematic Chain
# ----------------------------
joint7 = URDFLink(
    name="7_base_rotation",
    origin_translation=np.array([0, 0, 0.038]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 0, 1]),
    joint_type="revolute",
)

joint8 = URDFLink(
    name="8_shoulder_tilt",
    origin_translation=np.array([0, 0, 0.017]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 1, 0]),
    joint_type="revolute",
)

L2 = 0.110
joint9 = URDFLink(
    name="9_elbow_tilt",
    origin_translation=np.array([0, 0, L2]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, 1, 0]),
    joint_type="revolute",
)

L3 = 0.100
joint10 = URDFLink(
    name="10_wrist_tilt",
    origin_translation=np.array([L3, 0, 0]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([0, -1, 0]),
    joint_type="revolute",
)

L4 = 0.170
joint11 = URDFLink(
    name="11_gripper_rotate",
    origin_translation=np.array([0, 0, -L4]),
    origin_orientation=np.array([0, 0, 0]),
    rotation=np.array([1, 0, 0]),
    joint_type="revolute"
)

robot_chain = Chain(name="robot_chain_follower", links=[
    OriginLink(),
    joint7,
    joint8,
    joint9,
    joint10,
    joint11
])

# ----------------------------
# Dynamixel Setup for Follower
# ----------------------------
PROTOCOL_VERSION = 2.0
DXL_IDS = [7, 8, 9, 10, 11, 12]
BAUDRATE = 1e6
DEVICENAME = '/dev/tty.usbmodem58FD0170871'
TICKS_PER_REV = 4096


def rad_to_dxl(angle_rad):
    return int((angle_rad / (2 * np.pi)) * TICKS_PER_REV)


def dxl_to_rad(ticks):
    return (ticks / TICKS_PER_REV) * 2 * np.pi


port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)
if not port_handler.openPort():
    raise RuntimeError("Failed to open port for follower!")
if not port_handler.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baudrate for follower!")


def to_signed(val):
    if val > 0x7FFFFFFF:
        return val - 0x100000000
    return val


def to_unsigned(val):
    if val < 0:
        return val + 0x100000000
    return val


# Enable torque for follower servos (for the first 5 IDs)
TORQUE_ENABLE_ADDR = 64  # address may vary
TORQUE_ENABLE_VALUE = 1
for dxl_id in DXL_IDS[:5]:
    comm_result, error = packet_handler.write1ByteTxRx(
        port_handler, dxl_id, TORQUE_ENABLE_ADDR, TORQUE_ENABLE_VALUE)
    if comm_result != 0 or error != 0:
        print(
            f"Error enabling torque for servo {dxl_id}: {comm_result}, {error}")


def read_current_angles():
    angles = []
    for dxl_id in DXL_IDS[:5]:
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


def build_target_pose(x, y, z):
    pose = np.eye(4)
    pose[:3, 3] = np.array([x, y, z])
    return pose


def move_to_target_from_current(target_x_cm, target_y_cm, target_z_cm, current_angles=None):
    if current_angles is None:
        current_angles = read_current_angles()
    current_full_angles = np.concatenate(([0], current_angles))

    target_pose = build_target_pose(
        target_x_cm / 100.0, target_y_cm / 100.0, target_z_cm / 100.0)
    ik_solution = robot_chain.inverse_kinematics_frame(
        target_pose, initial_position=current_full_angles)
    desired_angles = ik_solution[1:6]

    for i, dxl_id in enumerate(DXL_IDS[:5]):
        target_tick = to_unsigned(rad_to_dxl(desired_angles[i]))

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, dxl_id, 116, target_tick)
        if comm_result != 0 or error != 0:
            print(
                f"Error sending command to servo {dxl_id}: {comm_result}, {error}")


if __name__ == "__main__":
    try:
        current_angles = read_current_angles()
        print("Current joint angles:", current_angles)
        print("\nRobot kinematic chain setup for follower arm is successful!")

        # Test move_to_target_from_current
        print("\nTesting move_to_target_from_current function...")
        move_to_target_from_current(18, 0, 25)  # Move to x=18cm, y=0cm, z=25cm
        print("Movement command sent successfully!")

    except Exception as e:
        print(f"Error during robot setup: {str(e)}")
