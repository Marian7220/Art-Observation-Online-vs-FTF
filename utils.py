# imports
import numpy as np
import os
import shutil
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

# global variables
## world & image sizes (width, height)
source_size = (1280, 720)
destination_size = (1000, 1000)  # img_sizes = [(670, 810), (810, 910), (900, 760), (770, 700), (800, 930)]


def get_surface_gaze_positions(gaze_positions_on_surface: str) -> np.ndarray:
    """
    This method calculates and returns gaze positions on surface.
    Parameters
    ----------
    gaze_positions_on_surface: str
        The absolute path of gaze positions on surface .csv file
    Returns
    -------
    surface_gaze_positions: np.ndarray
        Gaze positions on surface
    """
    gaze_positions_on_surface_df = pd.read_csv(gaze_positions_on_surface)
    frames = gaze_positions_on_surface_df["world_index"].max()
    min_frames = gaze_positions_on_surface_df["world_index"].min()
    surface_gaze_positions = np.zeros((frames, 2))
    for i in range(min_frames, frames):
        frame_rows = gaze_positions_on_surface_df.loc[gaze_positions_on_surface_df["world_index"] == i]
        surface_gaze_positions[i][0] = int(np.round(frame_rows["x_norm"].mean() * destination_size[0]))
        surface_gaze_positions[i][1] = destination_size[1] - int(
            np.round(frame_rows["y_norm"].mean() * destination_size[1]))
    return np.array(surface_gaze_positions, dtype=int)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros(4, dtype=int)

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = np.argmin(s)
    rect[2] = np.argmax(s)

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = np.argmin(diff)
    rect[3] = np.argmax(diff)

    # return the ordered coordinates
    return rect


def focus_video_on_surface(dir_name: str):
    """
    This method transforms a video such that it will focus on the given surface.
    Parameters
    ----------
    dir_name: str
        The directory containing the world video, the marker detections .csv file and the gaze positions on surface .csv file.
    """
    frames = []
    for file_name in os.listdir(dir_name):
        file_absolute_path = os.path.join(dir_name, file_name)
        if file_name == "world.mp4":
            world = file_absolute_path
        elif file_name == "gaze_positions_on_surface.csv":
            gaze_positions_on_surface = file_absolute_path
        elif file_name == "marker_detections.csv":
            marker_detections = file_absolute_path
    gaze_positions_on_surface = get_surface_gaze_positions(gaze_positions_on_surface)
    marker_detections_df = pd.read_csv(marker_detections)
    world_frames = cv.VideoCapture(world)
    frame_number = 0

    if not world_frames.isOpened():
        print("Error opening video stream or file")
    while world_frames.isOpened():
        ret, frame = world_frames.read()
        frame_marker_detections = np.float32(np.delete(
            marker_detections_df.loc[marker_detections_df["world_index"] == frame_number].to_numpy(), [0, 1], 1))
        if frame_number == gaze_positions_on_surface.shape[0]:
            break
        elif len(frame_marker_detections) != 4:
            frame_number += 1
            continue
        elif ret:
            pts = np.delete(frame_marker_detections, [2, 3, 4, 5, 6, 7], 1)
            order = order_points(pts)  # bl, br, tr, tl
            pts[order[0]] = frame_marker_detections[order[0], [6, 7]]
            pts[order[2]] = frame_marker_detections[order[2], [2, 3]]
            pts[order[1]] = frame_marker_detections[order[1], [4, 5]]
            frame_map = np.array(
                [[0, 0], [destination_size[0] - 1, 0], [destination_size[0] - 1, destination_size[1] - 1],
                 [0, destination_size[1] - 1]], np.float32)
            mat = cv.getPerspectiveTransform(pts[order], frame_map)
            frame = cv.warpPerspective(frame, mat, destination_size)
            if gaze_positions_on_surface[frame_number][0] >= 0 and gaze_positions_on_surface[frame_number][1] >= 0:
                frame = cv.circle(frame, gaze_positions_on_surface[frame_number], 1, (255, 0, 0), 20)
                frame = cv.circle(frame, gaze_positions_on_surface[frame_number], 1, (0, 255, 0), 5)
            frame_number += 1
            frames.append(frame)
        else:
            break
    world_frames.release()
    out = cv.VideoWriter(os.path.join(os.path.dirname(world), "surface_window") + ".mp4",
                         cv.VideoWriter_fourcc(*'mp4v'), 18.0,
                         (destination_size[0], destination_size[1]), 1)
    for im in frames:
        out.write(im)
    out.release()


def organize_data(absolute_path: str, depth=0):
    """
    This method organizes the data and readies it for use.
    Run on directory containing recordings in batches (directories)
    Parameters
    ----------
    absolute_path: int
        The absolute path of the directory to be organized
    depth : int
        For recursion purposes
    """
    for file_name in os.listdir(absolute_path):
        file_absolute_path = os.path.join(absolute_path, file_name)
        if os.path.isdir(file_absolute_path):
            if depth in [0, 1]:
                new_info_dir = file_absolute_path.replace("הקלטות - מוזיאון", "נתונים - מוזיאון")
                os.makedirs(new_info_dir)
                organize_data(file_absolute_path, depth + 1)
            if file_name == "exports" or file_name == "surfaces" or depth == 3:
                organize_data(file_absolute_path, depth + 1)
        elif os.path.isfile(file_absolute_path):
            if file_name == "world.mp4" and depth == 2:
                shutil.copyfile(file_absolute_path,
                                file_absolute_path.replace("הקלטות - מוזיאון", "נתונים - מוזיאון"))
            elif (file_name.startswith("marker_detections") or file_name.startswith(
                    "gaze_positions_on_surface")) and depth == 5:
                target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(absolute_path))).replace(
                    "הקלטות - מוזיאון", "נתונים - מוזיאון"), file_name)
                shutil.copyfile(file_absolute_path, target_path)
                if file_name.startswith("gaze_positions_on_surface"):
                    os.rename(target_path, target_path.replace("_Surface 1", ""))