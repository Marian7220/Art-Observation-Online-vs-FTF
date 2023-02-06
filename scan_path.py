import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path


def generate_scan_path(painting_im, subj=1, painting=1, first_scnds=False, seconds=0, screen=False, show_h=False):
    """
    Generates heat map for the given painting and subject
    :param painting_im: painting img (heatmap will be on top of this img)
    :param subj: no. of the subject (to locate it in the data)
    :param painting: no. of painting (to locate it's folder in the data)
    :param screen: screen: boolean, True if you want to use screen observation data, False if museum
    :param first_scnds: boolean, True if you want to take only x first seconds
    :param seconds: duration recording you want to consider in the generated scan path
    :param show_h: True if you want to see the generated scan path after saving
    :return: nothing, saves scan paths to the given path
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
    recording_location = Path("D:/CHBLD/sp_data") / directory / str(subj) / str(painting)
    surface_fixation_export = recording_location / "exports" / "001" / "surfaces" / "fixations_on_surface_Surface 1.csv"
    try:
        surface_df = pd.read_csv(surface_fixation_export)
    except:
        print("path not found")
        return

    cal = np.loadtxt(Path("D:/CHBLD/sp_data") / directory / str(subj) / "calibration (1).txt")
    if cal[painting - 1] == 0:
        title += "bad"
    elif cal[painting - 1] == 0.5:
        title += "okay"
    else:
        title += "good"

    # take only on surface points
    fixation_on_surf = surface_df[surface_df.on_surf == True]
    ini_ts = fixation_on_surf['world_timestamp'].iloc[0]
    fixation_on_surf.loc[:, 'world_timestamp'] -= ini_ts

    # take first seconds or the whole thing (30 sec)
    if first_scnds:
        fixation_on_surf = fixation_on_surf[fixation_on_surf.world_timestamp <= seconds]
    else:
        fixation_on_surf = fixation_on_surf[fixation_on_surf.world_timestamp <= 30]

    fixation_on_surf = fixation_on_surf.groupby(['fixation_id']).mean()

    # if there are no points left, return
    if fixation_on_surf.empty:
        print("there is no data! p" + str(subj) + " d" + str(painting))
        return

    # prepare data to plot later #######
    point_scale = fixation_on_surf["duration"]
    grid = painting_im.shape[0:2]  # height, width of the loaded image
    x = fixation_on_surf["norm_pos_x"]
    y = fixation_on_surf["norm_pos_y"]

    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at bottom left,
    # to the image coordinate system,
    # where the origin is at top left
    y = 1 - y

    x = np.asarray(x.values.tolist(), dtype='float64')
    y = np.asarray(y.values.tolist(), dtype='float64')

    # scale from [0,1] to img size
    x *= grid[0]
    y *= grid[0]

    # delete points out of borders
    point_scale = np.asarray(point_scale.values.tolist(), dtype='float64')
    # delete all points out of the image, x-axis
    y = y[x <= grid[1]]
    point_scale = point_scale[x <= grid[1]]
    x = x[x <= grid[1]]

    y = y[x >= 0]
    point_scale = point_scale[x >= 0]
    x = x[x >= 0]
    # delete all points out of the image, y-axis
    x = x[y <= grid[0]]
    point_scale = point_scale[y <= grid[0]]
    y = y[y <= grid[0]]

    x = x[y >= 0]
    point_scale = point_scale[y >= 0]
    y = y[y >= 0]

    # calculate distance between each 2 consecutive points - derivative - (to draw arrows later)
    dx, dy = [], []
    for i in range(len(x) - 1):
        dx.append(x[i+1]-x[i])
    for i in range(len(y) - 1):
        dy.append(y[i+1]-y[i])
    dx.append(0)
    dy.append(0)

    id_labels = list(range(1, y.shape[0]+1))

    # display reference image
    plt.figure(figsize=(20, 20))
    plt.title(title, fontsize=30)
    plt.imshow(painting_im, alpha=0.75)

    # display the lines and points for fixation
    polyline = plt.plot(x, y, "C3", lw=2)
    # use the duration to determine the scatter plot circle radius
    points = plt.scatter(x[:1], y[:1], s=point_scale[:1], alpha=0.5, color="cyan")
    points = plt.scatter(x[1:], y[1:], s=point_scale[1:], alpha=0.5)

    # draw arrows
    for i in range(len(x)):
        arrow = plt.arrow(x[i], y[i], dx[i], dy[i], width=2, length_includes_head=True, shape='full', head_width=20,
                          head_length=25, color='C3', fill=True)

    # numerate points
    ax = plt.gca()  # get plot current axes for annotation
    for i, l in enumerate(id_labels):
        ax.annotate(text=l, xy=(list(x)[i], list(y)[i]))

    plt.axis('off')     # don't show axis when you plot

    # save the result
    if first_scnds:
        name = '_scanpath'+str(seconds)+'.png'
    else:
        name = '_scanpath30.png'

    plt.savefig('D:/CHBLD/sp_data/' + directory +'/' + str(subj) + '/' + str(subj) + "_" + str(painting) + name, bbox_inches='tight')
    print("saved " + str(subj) + '_' + str(painting) + name)

    # show the result if wanted
    if show_h:
        plt.show()
    plt.close()

