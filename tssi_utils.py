import numpy as np
import pandas as pd
import re

from enum import IntEnum
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_LIPS, FACEMESH_RIGHT_EYEBROW, FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark
import tensorflow as tf
import math


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class FaceLandmark(IntEnum):
    LEFT_LOWER_EYEBROW_INNER = 285
    LEFT_LOWER_EYEBROW_OUTER = 276
    LEFT_UPPER_EYEBROW_INNER = 336
    LEFT_UPPER_EYEBROW_OUTER = 300
    RIGHT_LOWER_EYEBROW_INNER = 55
    RIGHT_LOWER_EYEBROW_OUTER = 46
    RIGHT_UPPER_EYEBROW_INNER = 107
    RIGHT_UPPER_EYEBROW_OUTER = 70
    LEFT_EYE_INNER = 362
    RIGHT_EYE_INNER = 133
    EXTERIOR_LIP = 0
    INFERIOR_LIP = 13


IGNORE_POSE_JOINTS = [
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,

    PoseLandmark.LEFT_PINKY,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,

    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX,

    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB
]

FILTERED_POSE_CONNECTIONS = [connection for connection in POSE_CONNECTIONS
                             if connection[0] not in IGNORE_POSE_JOINTS and
                             connection[1] not in IGNORE_POSE_JOINTS]

FILTERED_FACEMESH_CONNECTIONS = frozenset().union(*[
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW
])


def preprocess_dataframe(dataframe, with_root=True, with_midhip=True):
    x_columns = dataframe.columns[3::2]
    y_columns = dataframe.columns[4::2]
    xy_columns = dataframe.columns[3:]
    left_hand_columns = [
        column for column in dataframe.columns if "leftHand" in column]
    right_hand_columns = [
        column for column in dataframe.columns if "rightHand" in column]
    left_wrist_columns = ["pose_15_x", "pose_15_y"]
    right_wrist_columns = ["pose_16_x", "pose_16_y"]
    face_columns = [column for column in dataframe.columns if "face" in column]
    nose_columns = ["pose_0_x", "pose_0_y"]

    # Select xy columns
    selected_data = dataframe.loc[:, xy_columns]

    # Replace left hand columns with the left wrist coordinates
    no_left_hand_mask = np.all(selected_data[left_hand_columns].isna(), axis=1)
    selected_data.loc[no_left_hand_mask, left_hand_columns] = np.tile(
        selected_data.loc[no_left_hand_mask, left_wrist_columns], int(len(left_hand_columns) / 2))

    # Replace right hand columns with the right wrist coordinates
    no_right_hand_mask = np.all(
        selected_data[right_hand_columns].isna(), axis=1)
    selected_data.loc[no_right_hand_mask, right_hand_columns] = np.tile(
        selected_data.loc[no_right_hand_mask, right_wrist_columns], int(len(right_hand_columns) / 2))

    # Replace face columns with the nose coordinates
    no_face_mask = np.all(selected_data[face_columns].isna(), axis=1)
    selected_data.loc[no_face_mask, face_columns] = np.tile(
        selected_data.loc[no_face_mask, nose_columns], int(len(face_columns) / 2))

    # Move in the x-axis
    x_coordinate_smaller_than_0_mask = np.any(
        selected_data[x_columns] < 0, axis=1)
    x_offset = selected_data[x_coordinate_smaller_than_0_mask].min(
        axis=1).abs().values[:, np.newaxis]
    selected_data.loc[x_coordinate_smaller_than_0_mask,
                      x_columns] = selected_data.loc[x_coordinate_smaller_than_0_mask, x_columns] + x_offset

    # Move in the y-axis
    y_coordinate_smaller_than_0_mask = np.any(
        selected_data[y_columns] < 0, axis=1)
    y_offset = selected_data[y_coordinate_smaller_than_0_mask].min(
        axis=1).abs().values[:, np.newaxis]
    selected_data.loc[y_coordinate_smaller_than_0_mask,
                      y_columns] = selected_data.loc[y_coordinate_smaller_than_0_mask, y_columns] + y_offset

    # Scale data
    out_of_scale_mask = np.any(selected_data > 1, axis=1)
    out_of_scale_data = selected_data[out_of_scale_mask]
    scales = out_of_scale_data.max(axis=1).to_numpy()[:, np.newaxis]
    selected_data.loc[out_of_scale_mask, :] = out_of_scale_data / scales

    # Concat info
    info = dataframe.loc[:, ["video", "frame", "label"]].reset_index(drop=True)
    full_data = pd.concat([info, selected_data], axis=1)

    # Add root column
    if with_root:
        full_data['root_x'] = (full_data['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_x'] +
                               full_data['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_x']) / 2.
        full_data['root_y'] = (full_data['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_y'] +
                               full_data['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_y']) / 2.

    # Add midhip column
    if with_midhip:
        full_data['midhip_x'] = (full_data['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_x'] +
                                 full_data['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_x']) / 2.
        full_data['midhip_y'] = (full_data['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_y'] +
                                 full_data['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_y']) / 2.

    return full_data


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.adj = [[0 for i in range(self.num_nodes)]
                    for j in range(self.num_nodes)]

    # Function to add an edge to the graph
    # Considering a bidirectional edge
    def add_edge(self, start, end):
        start_idx = self.nodes.index(start)
        end_idx = self.nodes.index(end)
        self.adj[start_idx][end_idx] = 1
        self.adj[end_idx][start_idx] = 1

    # Function to remove an edge to the graph
    # Considering a bidirectional edge
    def remove_edge(self, start, end):
        start_idx = self.nodes.index(start)
        end_idx = self.nodes.index(end)
        self.adj[start_idx][end_idx] = 0
        self.adj[end_idx][start_idx] = 0

    # Function to add an edge to the graph by index
    # Considering a bidirectional edge
    def add_edge_by_index(self, start, end):
        self.adj[start][end] = 1
        self.adj[end][start] = 1

    # Function to remove an edge to the graph by index
    # Considering a bidirectional edge
    def remove_edge_by_index(self, start, end):
        self.adj[start][end] = 0
        self.adj[end][start] = 0

    def visit(self, start, visited):
        # Init path with the start node
        path = [start]

        # Set current node as visited
        visited[start] = True

        # For every node of the graph
        for i in range(self.num_nodes):
            if self.adj[start][i] == 1 and not visited[i]:
                path = path + self.visit(i, visited) + [start]

        return path

    # Function to perform DFS on the graph
    # Returns a list of nodes' indexes
    def dfs_by_index(self, start):
        paths = []
        visited = [False] * self.num_nodes

        while True:
            paths.append(self.visit(start, visited))
            if False in visited:
                start = visited.index(False)
            else:
                break

        return paths


def tssi_v1(columns):
    # Define joints
    joints = [col.replace("_x", "") for col in columns if "x" in col]

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FACEMESH_CONTOURS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # join the two lines of the left eyebrow
    graph.add_edge(
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_INNER)),
        "face_" + str(int(FaceLandmark.LEFT_UPPER_EYEBROW_INNER))
    )
    graph.add_edge(
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_OUTER)),
        "face_" + str(int(FaceLandmark.LEFT_UPPER_EYEBROW_OUTER))
    )

    # join the two lines of the right eyebrow
    graph.add_edge(
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER)),
        "face_" + str(int(FaceLandmark.RIGHT_UPPER_EYEBROW_INNER))
    )
    graph.add_edge(
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_OUTER)),
        "face_" + str(int(FaceLandmark.RIGHT_UPPER_EYEBROW_OUTER))
    )

    # join the exterior lip with the inferior lip
    graph.add_edge(
        "face_" + str(int(FaceLandmark.EXTERIOR_LIP)),
        "face_" + str(int(FaceLandmark.INFERIOR_LIP))
    )

    # join the nose with the left eye
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.LEFT_EYE_INNER))
    )

    # join the nose with the right eye
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.RIGHT_EYE_INNER))
    )

    # join the nose with the left eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_INNER))
    )

    # join the nose with the right eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER))
    )

    # join the nose with the exterior lip
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.EXTERIOR_LIP))
    )

    # join the nose with the pose's mouth left
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "pose_" + str(int(PoseLandmark.MOUTH_LEFT))
    )

    # join the nose with the pose's mouth right
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "pose_" + str(int(PoseLandmark.MOUTH_RIGHT))
    )

    # join the left wrist of the hand with the pose's left wrist
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.LEFT_WRIST)),
        "leftHand_" + str(int(HandLandmark.WRIST))
    )

    # join the right wrist of the hand with the pose's right wrist
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.RIGHT_WRIST)),
        "rightHand_" + str(int(HandLandmark.WRIST))
    )

    # Perform DFS starting at the nose
    start = graph.nodes.index("pose_0")
    paths = graph.dfs_by_index(start)
    tree_path = [graph.nodes[i] for path in paths[:-1] for i in path]

    # Generate column names
    x_sorted_columns = [joint + "_x" for joint in tree_path]
    y_sorted_columns = [joint + "_y" for joint in tree_path]

    # Debug info
    tmp = [
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)))),
        ("RIGHT HAND WRIST:", tree_path.index(
            "rightHand_" + str(int(HandLandmark.WRIST)))),
        ("MOUTH (EXTERIOR LIP):", tree_path.index("face_0")),
        ("MOUTH (INFERIOR LIP):", tree_path.index("face_13")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_EYE_INNER)))),
        ("RIGHT EYEBROW:", tree_path.index("face_55")),
        # ("RIGHT EYE:", tree_path.index("face_133")),
        ("LEFT EYEBROW:", tree_path.index("face_285")),
        ("LEFT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_EYE_INNER)))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        # ("LEFT EYE:", tree_path.index("face_362")),
        ("LEFT HAND WRIST:", tree_path.index(
            "leftHand_" + str(int(HandLandmark.WRIST)))),
    ]
    tmp.sort(key=lambda x: x[1])

    return tree_path, x_sorted_columns, y_sorted_columns, tmp, paths


def tssi_v3_legacy(columns):
    left_hand_columns = [column for column in columns if "leftHand" in column]
    right_hand_columns = [
        column for column in columns if "rightHand" in column]
    pose_columns = [column for column in columns if "pose" in column]

    # Define joints from right to left
    # face joints such as lips and eyebrows
    lips_joints = list(set(['face_' + str(item)
                       for connection in FACEMESH_LIPS for item in connection]))
    lips_joints.sort(key=natural_keys)
    right_eyebrow_joints = list(set(
        ['face_' + str(item) for connection in FACEMESH_RIGHT_EYEBROW for item in connection]))
    right_eyebrow_joints.sort(key=natural_keys)
    left_eyebrow_joints = list(set(
        ['face_' + str(item) for connection in FACEMESH_LEFT_EYEBROW for item in connection]))
    left_eyebrow_joints.sort(key=natural_keys)
    # pose joints such as eyes, nose and mouth
    pose_face_joints = [col.replace('_x', '') for col in pose_columns[::2]][:9]
    pose_face_joints.reverse()
    pose_body_joints = [col.replace('_x', '') for col in pose_columns[::2]][9:]
    pose_body_joints.reverse()
    # right and left hands joints
    right_hand_joints = [col.replace('_x', '')
                         for col in right_hand_columns[::2]]
    left_hand_joints = [col.replace('_x', '')
                        for col in left_hand_columns[::2]]
    joints = right_eyebrow_joints + left_eyebrow_joints + pose_face_joints + \
        lips_joints + pose_body_joints + right_hand_joints + left_hand_joints

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FILTERED_FACEMESH_CONNECTIONS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # join the two lines of the left eyebrow
    graph.add_edge(
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_INNER)),
        "face_" + str(int(FaceLandmark.LEFT_UPPER_EYEBROW_INNER))
    )
    graph.add_edge(
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_OUTER)),
        "face_" + str(int(FaceLandmark.LEFT_UPPER_EYEBROW_OUTER))
    )

    # join the two lines of the right eyebrow
    graph.add_edge(
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER)),
        "face_" + str(int(FaceLandmark.RIGHT_UPPER_EYEBROW_INNER))
    )
    graph.add_edge(
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_OUTER)),
        "face_" + str(int(FaceLandmark.RIGHT_UPPER_EYEBROW_OUTER))
    )

    # join the exterior lip with the inferior lip
    graph.add_edge(
        "face_" + str(int(FaceLandmark.EXTERIOR_LIP)),
        "face_" + str(int(FaceLandmark.INFERIOR_LIP))
    )

    # join the nose with the left eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_INNER))
    )

    # join the nose with the right eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER))
    )

    # join the nose with the exterior lip
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.EXTERIOR_LIP))
    )

    # join the left wrist of the hand with the pose's left wrist
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.LEFT_WRIST)),
        "leftHand_" + str(int(HandLandmark.WRIST))
    )

    # join the right wrist of the hand with the pose's right wrist
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.RIGHT_WRIST)),
        "rightHand_" + str(int(HandLandmark.WRIST))
    )

    # join the nose with the left shoulder
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "pose_" + str(int(PoseLandmark.LEFT_SHOULDER))
    )

    # join the nose with the right shoulder
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER))
    )

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)),
        "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER))
    )

    # REMOVE the connection between the left hip and the right hip
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.LEFT_HIP)),
        "pose_" + str(int(PoseLandmark.RIGHT_HIP))
    )

    # Perform DFS starting at the nose
    start = graph.nodes.index("pose_0")
    paths = graph.dfs_by_index(start)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Generate column names
    x_sorted_columns = [joint + "_x" for joint in tree_path]
    y_sorted_columns = [joint + "_y" for joint in tree_path]

    # Debug info
    tmp = [
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)))),
        ("RIGHT HAND WRIST:", tree_path.index(
            "rightHand_" + str(int(HandLandmark.WRIST)))),
        ("MOUTH (EXTERIOR LIP):", tree_path.index("face_0")),
        ("MOUTH (INFERIOR LIP):", tree_path.index("face_13")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_EYE_INNER)))),
        ("RIGHT EYEBROW:", tree_path.index("face_55")),
        # ("RIGHT EYE:", tree_path.index("face_133")),
        ("LEFT EYEBROW:", tree_path.index("face_285")),
        ("LEFT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_EYE_INNER)))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        # ("LEFT EYE:", tree_path.index("face_362")),
        ("LEFT HAND WRIST:", tree_path.index(
            "leftHand_" + str(int(HandLandmark.WRIST)))),
    ]
    tmp.sort(key=lambda x: x[1])

    return tree_path, x_sorted_columns, y_sorted_columns, tmp, paths


def tssi_v4(columns):
    left_hand_columns = [column for column in columns if "leftHand" in column]
    right_hand_columns = [
        column for column in columns if "rightHand" in column]
    pose_columns = [column for column in columns if "pose" in column]

    # Define joints from right to left
    # face joints such as lips and eyebrows
    lips_joints = list(set(['face_' + str(item)
                       for connection in FACEMESH_LIPS for item in connection]))
    lips_joints.sort(key=natural_keys)
    right_eyebrow_joints = list(set(
        ['face_' + str(item) for connection in FACEMESH_RIGHT_EYEBROW for item in connection]))
    right_eyebrow_joints.sort(key=natural_keys)
    left_eyebrow_joints = list(set(
        ['face_' + str(item) for connection in FACEMESH_LEFT_EYEBROW for item in connection]))
    left_eyebrow_joints.sort(key=natural_keys)
    # pose joints such as eyes, nose and mouth
    pose_face_joints = [col.replace('_x', '') for col in pose_columns[::2]][:9]
    pose_face_joints.reverse()
    pose_body_joints = [col.replace('_x', '') for col in pose_columns[::2]][9:]
    pose_body_joints.reverse()
    # right and left hands joints
    right_hand_joints = [col.replace('_x', '')
                         for col in right_hand_columns[::2]]
    left_hand_joints = [col.replace('_x', '')
                        for col in left_hand_columns[::2]]
    joints = ['root'] + right_eyebrow_joints + left_eyebrow_joints + pose_face_joints + \
        lips_joints + pose_body_joints + \
        right_hand_joints + left_hand_joints + ['midhip']

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FILTERED_FACEMESH_CONNECTIONS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # join the nose with the left eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.LEFT_LOWER_EYEBROW_INNER)))

    # join the nose with the right eyebrow
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER)))

    # join the nose with the inferior lip
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.NOSE)),
        "face_" + str(int(FaceLandmark.INFERIOR_LIP)))

    # join the left wrist of the pose to the left wrist of the hand
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.LEFT_WRIST)),
        "leftHand_" + str(int(HandLandmark.WRIST)))

    # join the right wrist of the pose to the right wrist of the hand
    graph.add_edge(
        "pose_" + str(int(PoseLandmark.RIGHT_WRIST)),
        "rightHand_" + str(int(HandLandmark.WRIST)))

    # join the ROOT with the left shoulder
    graph.add_edge(
        "root",
        "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)))

    # join the ROOT with the right shoulder
    graph.add_edge(
        "root",
        "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))

    # join the ROOT with the inferior lip
    graph.add_edge(
        "root",
        "face_" + str(int(FaceLandmark.INFERIOR_LIP)))

    # join the ROOT with the midhip
    graph.add_edge(
        "root",
        "midhip")

    # join the midhip with the left hip
    graph.add_edge(
        "midhip",
        "pose_" + str(int(PoseLandmark.LEFT_HIP)))

    # join the midhip with the right hip
    graph.add_edge(
        "midhip",
        "pose_" + str(int(PoseLandmark.RIGHT_HIP)))

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)),
        "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)))

    # REMOVE the connection between the left hip and the right hip
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.LEFT_HIP)),
        "pose_" + str(int(PoseLandmark.RIGHT_HIP)))

    # REMOVE the connection between the left shoulder and left hip
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)),
        "pose_" + str(int(PoseLandmark.LEFT_HIP)))

    # REMOVE the connection between the right shoulder and right hip
    graph.remove_edge(
        "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)),
        "pose_" + str(int(PoseLandmark.RIGHT_HIP)))

    # Perform DFS starting at the root
    root_index = graph.nodes.index("root")
    paths = graph.dfs_by_index(root_index)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Generate column names
    # x_sorted_columns = [joint + "_x" for joint in tree_path]
    # y_sorted_columns = [joint + "_y" for joint in tree_path]

    # Debug info
    info = [
        ("ROOT:", tree_path.index("root")),
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_SHOULDER)))),
        ("RIGHT HAND WRIST:", tree_path.index(
            "rightHand_" + str(int(HandLandmark.WRIST)))),
        # ("MOUTH (EXTERIOR LIP):", tree_path.index("face_0")),
        ("MOUTH (INFERIOR LIP):", tree_path.index("face_13")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.RIGHT_EYE_INNER)))),
        ("RIGHT EYEBROW:", tree_path.index("face_55")),
        # ("RIGHT EYE:", tree_path.index("face_133")),
        ("LEFT EYEBROW:", tree_path.index("face_285")),
        ("LEFT EYE INNER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_EYE_INNER)))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        # ("LEFT EYE:", tree_path.index("face_362")),
        ("LEFT HAND WRIST:", tree_path.index(
            "leftHand_" + str(int(HandLandmark.WRIST)))),
        ("MIDHIP:", tree_path.index("midhip")),
        ("LEFT HIP:", tree_path.index("pose_" + str(int(PoseLandmark.LEFT_HIP)))),
        ("RIGHT HIP:", tree_path.index("pose_" + str(int(PoseLandmark.RIGHT_HIP))))
    ]
    info.sort(key=lambda x: x[1])
    info

    return graph, tree_path


# Input = (0, 1) or (0, 255)


class PadIfLessThan(tf.keras.layers.Layer):
    def __init__(self, frames=128, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        height_pad = tf.math.maximum(0, self.frames - height)
        paddings = [[0, 0], [0, height_pad], [0, 0], [0, 0]]
        padded_images = tf.pad(images, paddings, "CONSTANT")
        return padded_images

class ResizeIfMoreThan(tf.keras.layers.Layer):
    def __init__(self, frames=128, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        new_size = [self.frames, width]
        resized = tf.cond(height > self.frames,
                          lambda: tf.image.resize(images, new_size),
                          lambda: images)
        return resized
    
# Input = (0, 1) or (0, 255)


class RandomSpeed(tf.keras.layers.Layer):
    def __init__(self, frames=128, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        p = tf.cast(0.75 * self.frames, tf.int32)
        x_min = tf.cond(height < p, lambda: height, lambda: p)
        x_max = self.frames + 1
        x = tf.random.uniform(shape=[], minval=x_min, maxval=x_max,
                              dtype=tf.int32, seed=self.seed)
        resized_images = tf.image.resize(images, [x, width])
        # paddings = [[0, 0], [0, self.frames - x], [0, 0], [0, 0]]
        # padded_images = tf.pad(resized_images, paddings, "CONSTANT")
        
        if self.debug:
            tf.print("speed", x)
        
        # return padded_images
        return resized_images

# Input = (0, 1) or (0, 255)

class RandomScale(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def round_down_float_to_1_decimal(self, num):
        return tf.math.floor(num * 10.0) / 10.0

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        red_maxs = tf.reduce_max(red, axis=-1, keepdims=True)
        red_mins = tf.reduce_min(red, axis=-1, keepdims=True)
        red_mids = (red_maxs + red_mins) / 2
        red_alphas_1 = (self.min_value - red_mids) / (red_mins - red_mids)
        red_alphas_2 = (self.max_value - red_mids) / (red_maxs - red_mids)
        red_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([red_alphas_1, red_alphas_2]))

        green_maxs = tf.reduce_max(green, axis=-1, keepdims=True)
        green_mins = tf.reduce_min(green, axis=-1, keepdims=True)
        green_mids = (green_maxs + green_mins) / 2
        green_alphas_1 = (self.min_value - green_mids) / (green_mins - green_mids)
        green_alphas_2 = (self.max_value - green_mids) / (green_maxs - green_mids)
        green_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([green_alphas_1, green_alphas_2]))

        max_alpha = tf.reduce_min([red_alpha, green_alpha])
        alpha = tf.random.uniform(shape=[], minval=0.5, maxval=max_alpha, seed=self.seed)
        new_red = alpha * (red - red_mids) + red_mids
        new_green = alpha * (green - green_mids) + green_mids
        
        if self.debug:
            tf.print("scale", alpha)

        return tf.stack([new_red, new_green, blue], axis=-1)

class RandomShift(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug
        
    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        left_offset = tf.reduce_min(red) - self.min_value
        right_offset = self.max_value - tf.reduce_max(red)
        red_shift = tf.random.uniform(shape=[],
                                      minval=tf.math.negative(left_offset),
                                      maxval=right_offset,
                                      seed=self.seed)
        
        if self.debug:
            tf.print("red shift", red_shift)

        bottom_offset = tf.reduce_min(green) - self.min_value
        top_offset = self.max_value - tf.reduce_max(green)
        green_shift = tf.random.uniform(shape=[],
                                        minval=tf.math.negative(bottom_offset),
                                        maxval=top_offset,
                                        seed=self.seed)

        new_red = tf.add(red, red_shift)
        new_green = tf.add(green, green_shift)
        
        if self.debug:
            tf.print("green shift", green_shift)

        return tf.stack([new_red, new_green, blue], axis=-1)
    
    # @tf.function
    # def call(self, images):
    #     return tf.vectorized_map(self.shift, images)
    

# Input = (0, 1) or (0, 255)


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, factor=45.0, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_degree = tf.math.negative(factor)
        self.max_degree = factor
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        degree = tf.random.uniform(shape=[],
                                   minval=self.min_degree,
                                   maxval=self.max_degree,
                                   seed=self.seed)
        if self.debug:
            tf.print("degree", degree)
        
        angle = degree * math.pi / 180.0
        
        [red, green, blue] = tf.unstack(image, axis=-1)
        mid_value = (self.max_value - self.min_value) / 2
        new_red = mid_value + \
            tf.math.cos(angle) * (red - mid_value) - \
            tf.math.sin(angle) * (green - mid_value)
        new_green = mid_value + \
            tf.math.sin(angle) * (red - mid_value) + \
            tf.math.cos(angle) * (green - mid_value)
        new_red = tf.clip_by_value(new_red, self.min_value, self.max_value)
        new_green = tf.clip_by_value(new_green, self.min_value, self.max_value)
        
        return tf.stack([new_red, new_green, blue], axis=-1)

# Input = (0, 1) or (0, 255)


class RandomFlip(tf.keras.layers.Layer):
    def __init__(self, mode, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        rand = tf.random.uniform(shape=[],
                                 minval=0.,
                                 maxval=1.,
                                 seed=self.seed)
        [red, green, blue] = tf.unstack(image, axis=-1)
        flip_horizontal = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'horizontal'))
        flip_vertical = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'vertical'))
        new_red = tf.cond(
            flip_horizontal, lambda: tf.add(-red, self.max_value), lambda: red)
        new_green = tf.cond(
            flip_vertical, lambda: tf.add(-green, self.max_value), lambda: green)
        
        if self.debug:
            tf.print("flip", rand)
        
        return tf.stack([new_red, new_green, blue], axis=-1)
