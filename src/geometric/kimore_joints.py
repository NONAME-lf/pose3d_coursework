# Standart indeces indexes of joints in Kinect v2
JOINTS = {
    'SpineBase':     0,
    'SpineMid':      1,
    'Neck':          2,
    'Head':          3,
    'ShoulderLeft':  4,
    'ElbowLeft':     5,
    'WristLeft':     6,
    'HandLeft':      7,
    'ShoulderRight': 8,
    'ElbowRight':    9,
    'WristRight':    10,
    'HandRight':     11,
    'HipLeft':       12,
    'KneeLeft':      13,
    'AnkleLeft':     14,
    'FootLeft':      15,
    'HipRight':      16,
    'KneeRight':     17,
    'AnkleRight':    18,
    'FootRight':     19,
    'SpineShoulder': 20,
    'HandTipLeft':   21,
    'ThumbLeft':     22,
    'HandTipRight':  23,
    'ThumbRight':    24,
}

# Number of joints in Kinect v2
NUM_JOINTS = 25


def get_joint(skeleton, name: str):
    if name not in JOINTS:
        raise KeyError(
            f"Невідома точка: {name!r}. Доступні: {sorted(JOINTS.keys())}"
        )
    return skeleton[..., JOINTS[name], :]
