import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
from numpy import linalg as LA
from shapely.geometry import Polygon, Point
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KNeighborsClassifier as KNearestNeighbors
from PIL import Image,ImageEnhance
import turtle

from PIL import Image, ImageDraw
from matplotlib.patches import Circle

from Utils import background_images, subjects_dict, input_fixations_directory, input_pupil_directory, \
    OFFSET_X, OFFSET_Y, height, width, SUBJECT_KEY, SUBJECT_ID, QUESTION_IDX

figure_counter = 0  # keep 0 please


# TODO: Removed bad and outdated analysis of Mean Pupil Histogram per Cluster. It is outdated in the sense that the
#  implementation changed too much that the old implementation was simply wrong.
#  If you want to re-implement it, it shouldn't prove to be too difficult anymore.


######################################################################################################################
################################################ Mode Selection ######################################################
# TODO: Idea, it might be a nice idea to encapsulate Mode Selection in a class in the future. However, for now this is
#   enough
# Select Clustering Mode (One of the modes has to be True)
# 1) First Mode and Parameters
USE_MEAN_SHIFT_CLUSTERING = True  # when True - clusters will be estimated from the data using MeanShift algorithm.
BANDWIDTH = 100.0


# 2) Second Mode and Parameters
USE_KNN_CLUSTERING = False  # when true - points will get labels using K Nearest neighbors
NEAREST_NEIGHBOR_K = 9

# 3) Third Mode and Parameters
USE_CUSTOM_CLUSTERING = False
ONLY_POLYGON_POINTS_RELEVANT = False
NUM_OF_AOI = 2  # How many Areas of Interests there are in the question. (Or, how many clusters there are) excluding
                # the outer AOI.
HARD_CODED = False  # if True you will use pre-existing clusters determined in HARD_CODED_CLUSTERS.
ENCODE_IN_REAL_TIME = False  # if True you can choose polygons to indicate AOIs directly from image in real time.
HARD_CODED_CLUSTERS = [
    [  # Question 1
        [[0, 1013], [0, 0], [540, 0], [540, 426], [500, 426], [500, 800], [400, 800], [400, 1013]],  # image
        [[465, 1013], [465, 840], [706, 840], [706, 1013]],  # legend
        [[1809, 1013], [1809, 324], [770, 324], [770,  554], [1060, 697], [1060, 815], [721, 815], [721,  1013]]  # text
    ],
    [  # Question 2
        [[0, 1155], [0, 0], [1266, 0], [1266, 1155]],  # image
        [[1266, 1155], [1266, 847], [2045, 847], [2045, 1155]]  # text
    ],
    [  # Question 3
        [[0, 1014], [1033, 1014],[1033, 470],[0, 470]],  # image
        [[1810, 1014], [1033, 1014], [1033, 512], [1810, 512]]  # text
    ],
    [  # Question 4
        [[0, 847], [0, 510], [800, 510], [800, 847]],  # question
        [[1518, 212], [650, 212], [649, 428], [800, 428], [800, 848], [1518, 848]]  # image
    ]
]

################################################ Mode Selection ######################################################
######################################################################################################################

# Draws a single polygon over the image 'img'
def drawPolygonOverImage(img, polygon):
    alpha = 0.5  # that's your transparency factor
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    overlay = img.copy()
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imshow("Polygon", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Allows to draw the clusters
def drawPolygonsOverImage(img, polygons):
    for polygon in polygons:
        drawPolygonOverImage(img, polygon)


# TODO: If the idea to replace Rectangles below is implemented you can just delete this class.
class Rect:
    def __init__(self, upper_left_x, upper_left_y, bottom_right_x, bottom_right_y):
        self.x1 = upper_left_x
        self.y1 = upper_left_y
        self.x2 = bottom_right_x
        self.y2 = bottom_right_y

    def is_point_inside(self, _x, _y):
        # Note: y2 < y1 cuz the way we output the data (y axis is flipped)
        return self.x1 <= _x <= self.x2 and self.y2 <= _y <= self.y1


# type in all the exculsions you would like
# Note: "height" and "width" will always refer to the question index being used. So whenever writing height or width,
# you will never make a mistake. (Unless ofcourse, you chose the wrong question index in the first place)
rectangles_to_exclude_question_1 = []
rectangles_to_exclude_question_2 = []
rectangles_to_exclude_question_3 = [Rect(0.0, height / 2.0, width, 0.0)]
rectangles_to_exclude_question_4 = []
rectangles_to_exclude = [
    rectangles_to_exclude_question_1,
    rectangles_to_exclude_question_2,
    rectangles_to_exclude_question_3,
    rectangles_to_exclude_question_4
]

# text colors - keep unchanged please:
CRED = '\033[91m'
CGREEN = '\33[32m'
CEND = '\033[0m'


# TODO: Idea: This could potentially be replaced by using Polygons (instead of rectangles) and allow for much better
#  flexibility.
def should_exclude_point(_x, _y, question_idx):
    for r in rectangles_to_exclude[question_idx]:
        if r.is_point_inside(_x, _y):
            return True
    return False


def isInside(circle_x, circle_y, rad, x, y):
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

def get_question_fixation_data():
    print("<Reading/Processing Fixation Data>:")

    data = []
    durations = {}
    xy_points_amount_per_subject = 0
    current_question_time_stamps = []
    for file_name in os.listdir(input_fixations_directory):
        # only consider the file that is relevant to our subject.
        if not file_name.startswith(SUBJECT_ID):
            continue
        file_directory = os.path.join(input_fixations_directory, file_name)
        df = pd.read_csv(file_directory)
        print(file_name)
        current_subject_times = subjects_dict[SUBJECT_KEY][QUESTION_IDX]  # tuple of (start, end) times per question_idx
        if current_subject_times is None:
            print(CRED + 'Error! Something is wrong with the start and end times.' + CEND)
            print(CRED + 'If the program continues it is likely yielding bad info.' + CEND)

        for i, point in enumerate(df.iterrows()):
            x, y = df['norm_pos_x'].iloc[i] * width, df['norm_pos_y'].iloc[i] * height
            if should_exclude_point(x, y, QUESTION_IDX):
                continue
            if   df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject_times[0] <= \
                    df['start_timestamp'].iloc[i] <= \
                    df['start_timestamp'].iloc[0] + current_subject_times[1]:
                duration = int(df['duration'].iloc[i] / 10)
                durations[(x, y)] = df['duration'].iloc[i]
                for j in range(duration):

                   data.append([df['start_timestamp'].iloc[i],
                            [df['norm_pos_x'].iloc[i] * width, (df['norm_pos_y'].iloc[i]) * height]])
                current_question_time_stamps.append(df['start_timestamp'])
        xy_points_amount_per_subject = len(data)
    total_duration = np.array(list(durations.values())).sum()
    print("<Reading/Processing Fixation Data Complete!>")
    return data, xy_points_amount_per_subject, durations, total_duration


def get_question_pupil_data(question_idx):
    print("<Reading/Processing Pupil Data>:")

    diameter_data = []
    for file in os.listdir(input_pupil_directory):
        if not file.startswith(SUBJECT_ID):
            continue
        diameter_data_current_subject = []
        file_directory = os.path.join(input_pupil_directory, file)
        df = pd.read_csv(file_directory)
        print(file)
        current_subject_times = subjects_dict[SUBJECT_KEY][question_idx]  # tuple of (start, end) times per question_idx
        if current_subject_times is None:
            continue
        for i, point in enumerate(df.iterrows()):

            # only include diameter data within the given questions' timestamp
            if df['pupil_timestamp'].iloc[0] + current_subject_times[0] <= \
                    df['pupil_timestamp'].iloc[i] <= \
                    df['pupil_timestamp'].iloc[0] + current_subject_times[1]:
                diameter_time = [df['pupil_timestamp'].iloc[i], df['diameter'].iloc[i]]
                diameter_data_current_subject.append(diameter_time)
        diameter_data.append(diameter_data_current_subject)
    print("<Reading/Processing Pupil Data Complete!>")
    return diameter_data


def get_question_data():
    print("####################################")
    print(f"Question {QUESTION_IDX + 1}:")
    data, xy_points_amount_per_subject, durations, total_duration = get_question_fixation_data()
    diameter_data = get_question_pupil_data(QUESTION_IDX)

    npDataArray = np.array(data, dtype=object)

    if npDataArray.size == 0:
        XY_TIME = None
        XY_ONLY = None
    else:
        XY_TIME = np.array([[t, x + OFFSET_X, y + OFFSET_Y] for [t, [x, y]] in npDataArray])
        XY_ONLY = np.array([[x, y] for t, x, y in XY_TIME])

    return XY_TIME, XY_ONLY, xy_points_amount_per_subject, diameter_data, durations, total_duration


def build_cluster_jump_matrix_between_different_clusters(xy_points_amount_per_subject, XY_ONLY, labels, n_clusters_):
    switch_mat = np.zeros([n_clusters_, n_clusters_])
    start = 0
    end = start + xy_points_amount_per_subject
    subject_data = XY_ONLY[start:end, :]
    last_label = -1
    for idx, label in enumerate(labels):
        if last_label != -1:
            if idx >= subject_data.shape[0]:
                break
            x, y = subject_data[idx, 0], subject_data[idx, 1]
            switch_mat[last_label, label] += 1.0
            switch_mat[last_label,last_label] = 0
            switch_mat[label,label] = 0
        last_label = label

    return switch_mat


def build_cluster_jump_array_within_same_cluster_and_disable_same_cluster_in_switch_matrix(switch_mat, n_clusters_):
    global figure_counter
    from matrixHeatMap import heatmapMatrix, annotate_heatmapMatrix
    total_jumps_within_each_area = [0 for _ in range(switch_mat.shape[0])]
    old_stuff = []
    for i in range(switch_mat.shape[0]):
        total_jumps_within_each_area[i] += (switch_mat[i][i])
        old_stuff.append(switch_mat[i][i])
        switch_mat[i][i] = 0
    plt.figure(figure_counter + 1)
    figure_counter += 1

    im, cbar = heatmapMatrix(switch_mat, np.arange(n_clusters_), np.arange(n_clusters_),
                             cmap="YlGn", cbarlabel="Frequency")
    plt.title("Transition Matrix Jumps from area i to j ")

    texts = annotate_heatmapMatrix(im, old_stuff)
    plt.savefig("Question" + str(QUESTION_IDX + 1) + "_Subject" + SUBJECT_KEY + "_TransitionMatrix")

    return switch_mat, total_jumps_within_each_area


def getAngle(three_angles):
    a, b, c = three_angles[0], three_angles[1], three_angles[2]
    import math
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def get_fixation_totalamount_variance_angles(XY_ONLY):
    angles = []
    for i in range(len(XY_ONLY) - 3):
        angles.append(getAngle(XY_ONLY[i:i + 3]))
    angles = np.array(angles)
    angles_hist = []
    angle_degrees_strings = []
    degrees = 5
    cap = 30
    for i in range(0, cap, degrees):
        angles_hist.append(len([angle for angle in angles if i <= angle < i + degrees]))
        angle_degrees_strings.append(str(i) + "-" + str(i + degrees))
    angles_hist.append(len(angles) - sum(angles_hist))
    angle_degrees_strings.append(str(cap) + "+")
    return len(XY_ONLY), XY_ONLY.std(), angles.mean(), angles_hist, angle_degrees_strings


def get_pupil_mean_variance(diameter_data):
    flat_list = [item for sublist in diameter_data for item in sublist]
    return np.mean(flat_list), np.std(flat_list)


def get_duration_per_cluster(XY_ONLY, durations, labels, n_clusters_):
    durations_per_visit_per_cluster = [[] for cluster in range(n_clusters_)]
    old_x, old_y = -1, -1
    idx = -1
    total_stay = 0
    last_cluster = -1
    for x, y in XY_ONLY:
        idx += 1
        if x == old_x and y == old_y:
            continue
        old_x, old_y = x, y
        cluster = labels[idx]
        if cluster != last_cluster:
            if total_stay > 0:
                durations_per_visit_per_cluster[last_cluster].append(total_stay)
                total_stay = 0
            last_cluster = cluster
        duration = durations[(x, y)]
        total_stay += duration
    durations_per_visit_per_cluster[last_cluster].append(total_stay)

    stay_duration_mean_per_cluster = [np.array(duration_list_in_cluster).mean() for duration_list_in_cluster in
                                      durations_per_visit_per_cluster]
    return stay_duration_mean_per_cluster


AOI = []


def click_event(event, x, y, flags, params):
    global AOI
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, ' ', height - y, height)
        AOI.append([x, height - y])


def vis_covariance(XY_ONLY):

    max_x = width
    max_y = height

    cov_x = np.array([x for (x, y) in XY_ONLY])
    cov_x = cov_x / max_x

    cov_y = np.array([y for (x, y) in XY_ONLY])
    cov_y = cov_y / max_y

    cov = np.array([cov_x, cov_y])
    cov_matrix = np.cov(cov, bias=False)
    print(cov_matrix)
    eigenvalues, eigenvectors = LA.eig(cov_matrix)
    np.set_printoptions(precision=5)
    print("eigenvalues:\n" + str(eigenvalues))
    # print("eigenvectors:\n" + str(eigenvectors))
    print("This was Question " + str(QUESTION_IDX + 1))


def keep_only_points_in_polygons(XY_ONLY, polygons):
    NEW_XY_ONLY = []
    for coordinate in XY_ONLY:
        point = Point(coordinate[0], coordinate[1])  # COME BACK HERE ELIE
        for i in range(NUM_OF_AOI):
            if polygons[i].contains(point):
                NEW_XY_ONLY.append([coordinate[0], coordinate[1]])
    return np.array(NEW_XY_ONLY)


def keep_only_points_in_circles(XY_ONLY):
    NEW_XY_ONLY = []
    for x,y in XY_ONLY:
        if isInside(320, 530, 200, x, y) or isInside(272, 160, 200, x, y) or isInside(400, 878, 50, x, y)\
                or isInside(775, 890, 100, x, y) or isInside(570, 865, 90, x, y) or isInside(725, 330, 70, x, y) or \
                isInside(106, 795, 120, x, y):
            NEW_XY_ONLY.append([x,y])
    return np.array(NEW_XY_ONLY)


def constructClusters(XY_ONLY):
    # Return Parameters:
    labels, n_clusters_, cluster_centers = None, None, None

    ##################################

    if USE_MEAN_SHIFT_CLUSTERING:
        clustering = MeanShift(bandwidth=BANDWIDTH, max_iter=300, n_jobs=2, bin_seeding=True).fit(XY_ONLY)
        cluster_centers = clustering.cluster_centers_
        labels = clustering.labels_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

    elif USE_KNN_CLUSTERING:
        KNN = KNearestNeighbors(n_neighbors=NEAREST_NEIGHBOR_K)
        df = pd.read_csv('clusters.csv')
        df_centers = pd.read_csv('cluster_centers.csv')
        loaded_data = [[df['x'].iloc[i], df['y'].iloc[i]] for i in range(len(df['x']))]
        loaded_labels = [df['labels'].iloc[i] for i in range(len(df['labels']))]
        cluster_centers = [[df_centers['cluster_centers_x'].iloc[i], df_centers['cluster_centers_y'].iloc[i]] for i
                           in range(len(df_centers['cluster_centers_x']))]
        KNN.fit(loaded_data, loaded_labels)
        labels = KNN.predict(XY_ONLY)
        labels_unique = np.unique(loaded_labels)
        n_clusters_ = len(labels_unique)

    elif USE_CUSTOM_CLUSTERING:
        global AOI
        img = cv2.imread(os.path.join('Heatmap', background_images[QUESTION_IDX]), 1)
        CLUSTERS = []
        labels = []
        n_clusters_ = NUM_OF_AOI + 1
        polygons = []

        if HARD_CODED:
            #   Convert AOI to Polygon
            for i in range(NUM_OF_AOI):
                xy = HARD_CODED_CLUSTERS[QUESTION_IDX][i]
                for indexN in range(len(xy)):
                    xy[indexN][1] = height - xy[indexN][1]
                    # xy[indexN][1] = height - xy[indexN][1]  # comment for good images of polygons
                polygons.append(Polygon(xy))

        elif ENCODE_IN_REAL_TIME:
            for i in range(NUM_OF_AOI):
                print("i=" + str(i))
                AOI = []
                cv2.imshow('image', img)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                CLUSTERS.append(AOI)

            #   Convert AOI to Polygon
            for i in range(NUM_OF_AOI):
                polygons.append(Polygon(CLUSTERS[i]))
        else:
            print(CRED + 'Error! Did not input proper Clustering Case. The program will fail or output illogical data.' + CEND)
            print(CRED + 'You mispicked HARD_CODED and ENCODE_IN_REAL_TIME.' + CEND)

        if ONLY_POLYGON_POINTS_RELEVANT:
            XY_ONLY = keep_only_points_in_polygons(XY_ONLY, polygons)
        #   Matching labels to each point
        for coordinate in XY_ONLY:
            point = Point(coordinate[0], coordinate[1])  # COME BACK HERE ELIE
            for i in range(NUM_OF_AOI):
                if polygons[i].contains(point):
                    labels.append(i)
                    break
                if i == NUM_OF_AOI - 1:
                    labels.append(NUM_OF_AOI)
    else:
        print(CRED + 'Error! Did not input proper Clustering Case. The program will fail or output illogical data.' + CEND)
        print(CRED + 'You mispicked the Mode Selection.' + CEND)

    return labels, n_clusters_, cluster_centers

def drow_lines(switch_mat , cluster_centers ) :
    array_flag = [0 , 0 , 0, 0, 0, 0, 0,0]
    array = [[0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0, 0, 0, 0, 0, 0,0,0],
             [0, 0, 0, 0, 0, 0,0,0],
             [0, 0, 0, 0, 0, 0,0,0]
             ]
    for i in range(switch_mat.shape[0]):
        for j in range(switch_mat.shape[1]):
            if (switch_mat[i][j] != 0 ):
                if(( switch_mat[j][i] != 0  and ( array[i][j] == 0 )) ) :
                    array[i][j] = array[j][i] = 1
                    cluster_center = cluster_centers[i]
                    cluster_center_j = cluster_centers[j]

                    plt.arrow(cluster_center[0]+40 ,cluster_center[1] +10,  cluster_center_j[0] - cluster_center[0] +1 ,
                                  cluster_center_j[1] - cluster_center[1] +1 , width=switch_mat[i][j] * 7, length_includes_head=True, shape='full',
                                  head_width= 40 ,
                                  head_length=40, color='blue', fill=True)
                else :
                    cluster_center = cluster_centers[i]
                    cluster_center_j = cluster_centers[j]
                    plt.arrow(cluster_center[0] , cluster_center[1] , cluster_center_j[0] - cluster_center[0] + 1,
                          cluster_center_j[1] - cluster_center[1] + 1, width=switch_mat[i][j] * 7,
                          length_includes_head=True, shape='full',
                          head_width=40,
                          head_length=40, color='yellow', fill=True)



def run_analysis():
    global figure_counter
    figure_counter = 0  # keep 0 please

    # extract question data
    XY_TIME, XY_ONLY, xy_points_amount_per_subject, diameter_data, durations, total_duration = \
        get_question_data()
    if XY_TIME is None:  # if empty
        print(CRED + "Skipping - No data provided for current question" + CEND)
        pass
    #    print('XY_TIME Shape: ' + str(XY_TIME.shape))
    #   print('XY_ONLY Shape: ' + str(XY_ONLY.shape))
    #XY_ONLY =  keep_only_points_in_circles(XY_ONLY)
    # create clusters and extract cluster data
    labels, n_clusters_, cluster_centers = constructClusters(XY_ONLY)

    print(CGREEN + "Finish clustering" + CEND)
    print(xy_points_amount_per_subject)
    switch_mat = build_cluster_jump_matrix_between_different_clusters(xy_points_amount_per_subject, XY_ONLY,
                                                                           labels, n_clusters_)
    switch_mat, total_jumps_within_each_area = \
        build_cluster_jump_array_within_same_cluster_and_disable_same_cluster_in_switch_matrix(switch_mat, n_clusters_)
    total_number_of_fixations, fixation_variance, angles_mean, angles_hist, angle_degrees_strings = \
        get_fixation_totalamount_variance_angles(XY_ONLY)
    mean_pupil_size, variance_pupil_size = get_pupil_mean_variance(diameter_data)
    # stay_duration_mean_per_cluster = get_duration_per_cluster(XY_ONLY, durations, labels, n_clusters_)


    # Plot result
    import matplotlib.patches as mpatches
    from itertools import cycle
    plt.figure(figure_counter + 1)
    figure_counter += 1
    plt.clf()

    hist = []
    hist_color = []
    image_path = os.path.join('Heatmap', background_images[QUESTION_IDX])
    img = Image.open(image_path)
    img = ImageOps.flip(img)
    img = ImageEnhance.Contrast(img)
    factor = 0.4
    img = img.enhance(factor)


    legend = []
    # colors = cycle('bgrcmykwbgrcmykbgrcmykbgrcmyk')
    colors = cycle(
        ['#dfdac4', '#00baff', '#7c31f2', '#00e5a3', '#ff1095','red',
         '#eaf27c', '#b13034', '#c86409', '#188fa7', '#785589'])
    # time_per_cluster = []

    for k, col in zip(range(n_clusters_), colors):
        my_members = [i for i, x in enumerate(labels) if x == k]
        # time_for_each_visit_in_current_cluster = get_every_visit_time(my_members, T)
        # time_per_cluster.append(time_for_each_visit_in_current_cluster)
        hist.append(len(my_members))
        hist_color.append(col)
        XY_ONLY = np.array([[x, y] for (x, y) in XY_ONLY])
        plt.scatter(XY_ONLY[my_members, 0], XY_ONLY[my_members, 1], c=col, marker='.')
        patch = mpatches.Patch(color=col, label=k)
        legend.append(patch)
        if USE_MEAN_SHIFT_CLUSTERING or USE_KNN_CLUSTERING:
            cluster_center = cluster_centers[k]
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k',
                     markersize=10)

    drow_lines(switch_mat, cluster_centers)
   # plt.legend(handles=legend )
    plt.title(f'Painting 1 - Estimated number of clusters: {n_clusters_} , Points & Arrows')
    plt.imshow(img, origin='lower')



    # turn to probablistic histogram
    hist = [x / sum(hist) for x in hist]

    # draw cluster histogram
    plt.figure(figure_counter + 1)
    figure_counter += 1
    plt.title(f"Painting 1 - Cluster Histogram")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.legend(handles=legend)
    y_pos = np.arange(len(hist_color))
    plt.bar(y_pos, hist, color=hist_color)
    plt.xticks(y_pos, np.arange(len(hist_color)))


    # draw stay duration mean per cluster
    # uncommnet later
    # plt.figure(figure_counter + 1)
    # figure_counter += 1
    # plt.title(f"Question {question_idx + 1} - Duration mean per cluster")
    # plt.xlabel("Cluster")
    # plt.ylabel("Mean Duration Stay (m/secs)")
    # y_pos = np.arange(len(stay_duration_mean_per_cluster))
    # plt.bar(y_pos, stay_duration_mean_per_cluster, color=hist_color)
    # plt.xticks(y_pos, np.arange(len(stay_duration_mean_per_cluster)))


    plt.show()
    drow_lines(switch_mat, cluster_centers)

    plt.title(f'Painting 1 - Transitions Between Areas')
    plt.imshow(img , origin='lower')
    plt.show()



run_analysis()
print("####################################")
print(CGREEN + "Analysis presentations complete." + CEND)
