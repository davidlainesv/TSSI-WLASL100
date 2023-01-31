import tensorflow as tf
import numpy as np
import pandas as pd
from mediapipe.python.solutions.pose import PoseLandmark


def preprocess_dataframe(dataframe, with_root=True, with_midhip=False):
    x_columns = dataframe.columns[3::2]
    y_columns = dataframe.columns[4::2]
    xy_columns = dataframe.columns[3:]
    left_hand_columns = [
        column for column in dataframe.columns if "leftHand" in column]
    right_hand_columns = [
        column for column in dataframe.columns if "rightHand" in column]
    left_wrist_columns = ['leftHand_' + str(int(PoseLandmark.LEFT_WRIST)) +
                          '_x', 'leftHand_' + str(int(PoseLandmark.LEFT_WRIST)) + '_y']
    right_wrist_columns = ['rightHand_' + str(int(PoseLandmark.RIGHT_WRIST)) +
                           '_x', 'rightHand_' + str(int(PoseLandmark.RIGHT_WRIST)) + '_y']
    face_columns = [column for column in dataframe.columns if "face" in column]
    nose_columns = ["pose_0_x", "pose_0_y"]

    # Add root column
    if with_root:
        dataframe['root_x'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_x'] +
                               dataframe['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_x']) / 2.
        dataframe['root_y'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_y'] +
                               dataframe['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_y']) / 2.

    # Add midhip column
    if with_midhip:
        dataframe['midhip_x'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_x'] +
                                 dataframe['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_x']) / 2.
        dataframe['midhip_y'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_y'] +
                                 dataframe['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_y']) / 2.

    # Select xy columns
    centered_data = dataframe.copy()
    # centered_data[x_columns] = centered_data[x_columns] - \
    #     dataframe['root_x'].to_numpy()[:, np.newaxis]
    # centered_data[y_columns] = centered_data[y_columns] - \
    #     dataframe['root_y'].to_numpy()[:, np.newaxis]

    # Replace left hand columns with the left wrist coordinates
    no_left_hand_mask = np.all(centered_data[left_hand_columns].isna(), axis=1)
    centered_data.loc[no_left_hand_mask, left_hand_columns] = np.tile(
        centered_data.loc[no_left_hand_mask, left_wrist_columns], int(len(left_hand_columns) / 2))

    # Replace right hand columns with the right wrist coordinates
    no_right_hand_mask = np.all(
        centered_data[right_hand_columns].isna(), axis=1)
    centered_data.loc[no_right_hand_mask, right_hand_columns] = np.tile(
        centered_data.loc[no_right_hand_mask, right_wrist_columns], int(len(right_hand_columns) / 2))

    # Replace face columns with the nose coordinates
    no_face_mask = np.all(centered_data[face_columns].isna(), axis=1)
    centered_data.loc[no_face_mask, face_columns] = np.tile(
        centered_data.loc[no_face_mask, nose_columns], int(len(face_columns) / 2))

    # Normalize data
    repetitions = centered_data.groupby("video").size()
    max_per_video = centered_data.abs().groupby(
        "video")[xy_columns].max().max(axis=1)
    max_per_video_repeated = max_per_video.repeat(
        repetitions).to_numpy()[:, np.newaxis]
    normalized_data = centered_data.copy()
    normalized_data[xy_columns] = normalized_data[xy_columns] / \
        max_per_video_repeated

    return normalized_data


def preprocess_dataframe_from0_to_1(dataframe, with_root=True, with_midhip=False):
    x_columns = dataframe.columns[3::2]
    y_columns = dataframe.columns[4::2]
    xy_columns = dataframe.columns[3:]
    left_hand_columns = [
        column for column in dataframe.columns if "leftHand" in column]
    right_hand_columns = [
        column for column in dataframe.columns if "rightHand" in column]
    left_wrist_columns = ['leftHand_' + str(int(PoseLandmark.LEFT_WRIST)) +
                          '_x', 'leftHand_' + str(int(PoseLandmark.LEFT_WRIST)) + '_y']
    right_wrist_columns = ['rightHand_' + str(int(PoseLandmark.RIGHT_WRIST)) +
                           '_x', 'rightHand_' + str(int(PoseLandmark.RIGHT_WRIST)) + '_y']
    face_columns = [column for column in dataframe.columns if "face" in column]
    nose_columns = ["pose_0_x", "pose_0_y"]

    # Add root column
    if with_root:
        dataframe['root_x'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_x'] +
                               dataframe['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_x']) / 2.
        dataframe['root_y'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_SHOULDER)) + '_y'] +
                               dataframe['pose_' + str(int(PoseLandmark.RIGHT_SHOULDER)) + '_y']) / 2.

    # Add midhip column
    if with_midhip:
        dataframe['midhip_x'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_x'] +
                                 dataframe['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_x']) / 2.
        dataframe['midhip_y'] = (dataframe['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_y'] +
                                 dataframe['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_y']) / 2.

    # Select xy columns
    centered_data = dataframe.copy()
    centered_data[x_columns] = centered_data[x_columns] - \
        dataframe['root_x'].to_numpy()[:, np.newaxis]
    centered_data[y_columns] = centered_data[y_columns] - \
        dataframe['root_y'].to_numpy()[:, np.newaxis]

    # Replace left hand columns with the left wrist coordinates
    no_left_hand_mask = np.all(centered_data[left_hand_columns].isna(), axis=1)
    centered_data.loc[no_left_hand_mask, left_hand_columns] = np.tile(
        centered_data.loc[no_left_hand_mask, left_wrist_columns], int(len(left_hand_columns) / 2))

    # Replace right hand columns with the right wrist coordinates
    no_right_hand_mask = np.all(
        centered_data[right_hand_columns].isna(), axis=1)
    centered_data.loc[no_right_hand_mask, right_hand_columns] = np.tile(
        centered_data.loc[no_right_hand_mask, right_wrist_columns], int(len(right_hand_columns) / 2))

    # Replace face columns with the nose coordinates
    no_face_mask = np.all(centered_data[face_columns].isna(), axis=1)
    centered_data.loc[no_face_mask, face_columns] = np.tile(
        centered_data.loc[no_face_mask, nose_columns], int(len(face_columns) / 2))

    # Move in the x-axis
    x_coordinate_smaller_than_0_mask = np.any(
        centered_data[x_columns] < 0, axis=1)
    x_offset = centered_data[x_coordinate_smaller_than_0_mask][x_columns].min(
        axis=1).abs().values[:, np.newaxis]
    centered_data.loc[x_coordinate_smaller_than_0_mask,
                      x_columns] = centered_data.loc[x_coordinate_smaller_than_0_mask, x_columns] + x_offset

    # Move in the y-axis
    y_coordinate_smaller_than_0_mask = np.any(
        centered_data[y_columns] < 0, axis=1)
    y_offset = centered_data[y_coordinate_smaller_than_0_mask][y_columns].min(
        axis=1).abs().values[:, np.newaxis]
    centered_data.loc[y_coordinate_smaller_than_0_mask,
                      y_columns] = centered_data.loc[y_coordinate_smaller_than_0_mask, y_columns] + y_offset

    # Normalize data
    repetitions = centered_data.groupby("video").size()
    max_per_video = centered_data.abs().groupby(
        "video")[xy_columns].max().max(axis=1)
    max_per_video_repeated = max_per_video.repeat(
        repetitions).to_numpy()[:, np.newaxis]
    normalized_data = centered_data.copy()
    normalized_data[xy_columns] = normalized_data[xy_columns] / \
        max_per_video_repeated

    return normalized_data


def preprocess_dataframe_legacy(dataframe, with_root=True, with_midhip=True):
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
        full_data['midhip_y'] = (full_data['pose_' + str(int(PoseLandmark.LEFT_HIP)) + '_y'] +
                                 full_data['pose_' + str(int(PoseLandmark.RIGHT_HIP)) + '_y']) / 2.

    return full_data


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
