import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter


def generate_heatmap(painting_im, subj=1, painting=1, screen=False, show_h=False):
    """
    Generates heat map for the given painting and subject
    :param painting_im: painting img (heatmap will be on top of this img)
    :param subj: no. of the subject (to locate it in the data)
    :param painting: no. of painting (to locate it's folder in the data)
    :param screen: screen: boolean, True if you want to use screen observation data, False if museum
    :param show_h: True if you want to see the generated heatmap after saving
    :return: nothing, saves heatmaps to the given path
    to use you can change paths of data ("directory", "name" and the path passed to read_csv, loadtxt
     and savefig)
    """
    # load data
    if screen:
        directory = "data_c"
        title = "screen: "
    else:
        directory = "data_m"
        title = "museum: "
    recording_location = Path("D:/CHBLD/hm_data") / directory / str(subj) / str(painting)
    surface_gaze_export = recording_location / "gaze_positions_on_surface.csv"

    # if there is no data / no file
    try:
        surface_df = pd.read_csv(surface_gaze_export)
    except:
        print("path not found")
        return

    gaze_on_surf = surface_df[(surface_df.on_surf == True) & (surface_df.confidence > 0.8)]
    if gaze_on_surf.empty:
        print("there is no data! p" + str(subj) + " d" + str(painting))
        return

    # load txt file containing rating of calibration, ratings are 0, 0.5, 1. zero is lowest.
    cal = np.loadtxt(Path("D:/CHBLD/hm_data") / directory / str(subj) / "calibration (1).txt")
    if cal[painting - 1] == 0:
        title += "bad"
    elif cal[painting - 1] == 0.5:
        title += "okay"
    else:
        title += "good"

    grid = painting_im.shape[0:2]   # height, width of the loaded image
    heatmap_detail = 0.035   # this will determine the gaussian blur kernel of the image (higher number = more blur)

    gaze_on_surf_x = gaze_on_surf['x_norm']
    gaze_on_surf_y = gaze_on_surf['y_norm']

    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at bottom left,
    # to the image coordinate system,
    # where the origin is at top left
    gaze_on_surf_y = 1 - gaze_on_surf_y

    # make the histogram
    hist, x_edges, y_edges = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1.0], [0, 1.0]],
        normed=False,
        bins=grid
    )

    # normalize - using log.
    hist += 1
    hist = np.log(hist) * 1000
    if np.amax(hist) != 0:
        hist *= (1000/(np.amax(hist)))

    # gaussian blur kernel as a function of grid/surface size
    filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
    heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)

    # display the histogram and reference image
    plt.figure(figsize=(20, 20))
    plt.title(title, fontsize=30)
    plt.imshow(painting_im)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig('D:/CHBLD/hm_data/' + directory +'/' + str(subj) + '/' + str(subj) + '_' + str(painting) +
                '_heatmap.png', bbox_inches='tight')
    print('s' + str(subj) + ' ' + str(painting) + '_heatmap.png saved.')
    if show_h:
        plt.show()
    plt.close()
