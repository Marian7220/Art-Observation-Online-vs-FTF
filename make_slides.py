import matplotlib.pyplot as plt


def make_slides_sp(painting, first_scnds=False, seconds=0, screen=False):
    """
    make slides containing subjects scan paths for each painting.
    :param painting: painting no. (to find it in the data)
    :param first_scnds: boolean, True if you want to take only x first seconds
    :param seconds: duration of scan paths you want to put in the slides
    :param screen: boolean, True if you want to use screen observation data, False if museum
    :return: nothing, it saves image slides to the given path
    to use you can change no. of "sbjcts", "paintings" and change paths of data
    ("directory", "name" and the path passed to imread and savefig)
    """
    if screen:
        directory = "data_c"
    else:
        directory = "data_m"

    if first_scnds:
        name = '_scanpath'+str(seconds)+'.png'
    else:
        name = '_scanpath30.png'
    c = 0
    valid = -1
    sbjcts = 22
    paintings = 5
    for i in range(sbjcts):
        try:
            im = plt.imread('D:/CHBLD/sp_data/'+directory+'/' + str(i+1) + '/' + str(i+1) + "_" + str(painting) + name)
            valid += 1
        except:
            print("path not found")
            continue
        if valid % (paintings-1) == 0:
            plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, (valid % (paintings-1))+1)
        plt.imshow(im)
        plt.axis('off')
        if (valid+1) % (paintings-1) == 0 or i + 1 == sbjcts:
            plt.savefig('D:/CHBLD/sp_data/'+directory+'/sp_'+str(painting) + '_slide' + str(c) + name, bbox_inches='tight')
            plt.close()
            c += 1
        print((valid % (paintings-1))+1)


def make_slides_hm(painting, screen=False):
    """
    make slides containing subjects heatmaps for each painting.
    :param painting: painting no. (to find it in the data)
    :param screen: boolean, True if you want to use screen observation data, False if museum
    :return: nothing, it saves image slides to the given path
    to use you can change no. of "sbjcts", "paintings" and change paths of data
    ("directory", "name" and the path passed to imread and savefig)
    """

    name = '_heatmap.png'
    if screen:
        directory = "data_c"
    else:
        directory = "data_m"
    c = 0
    valid = -1
    sbjcts = 22
    paintings = 5
    for i in range(sbjcts):
        try:
            im = plt.imread('D:/CHBLD/hm_data/'+directory+'/' + str(i+1) + '/' + str(i+1) + "_" + str(painting) + name)
            valid += 1
        except:
            print("path not found")
            continue
        if valid % (paintings-1) == 0:
            plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, (valid % (paintings-1))+1)
        plt.imshow(im)
        plt.axis('off')
        if (valid+1) % (paintings-1) == 0 or i + 1 == sbjcts:
            plt.savefig('D:/CHBLD/hm_data/'+directory+'/hm_'+str(painting) + '_slide' + str(c) + name, bbox_inches='tight')
            plt.close()
            c += 1
        print((valid % (paintings-1))+1)


def make_slides_subject_sp(sub_n, first_scnds=False, seconds=0, screen=False):
    """
    make slides containing paintings scan paths for each subject.
    :param sub_n: subject no. (to find it in the data)
    :param first_scnds: boolean, True if you want to take only x first seconds
    :param seconds: duration of scan paths you want to put in the slides
    :param screen: boolean, True if you want to use screen observation data, False if museum
    :return: nothing, it saves image slides to the given path
    to use you can change no. of "paintings" and change paths of data
    ("directory", "name" and the path passed to imread and savefig)
    """

    if screen:
        directory = "data_c"
    else:
        directory = "data_m"

    if first_scnds:
        name = '_scanpath'+str(seconds)+'.png'
    else:
        name = '_scanpath30.png'
    valid = -1
    paintings = 5
    for i in range(paintings):
        try:
            im = plt.imread('D:/CHBLD/sp_data/'+directory+'/' + str(sub_n) + '/' + str(sub_n) + "_" + str(i+1) + name)
            valid += 1
        except:
            print("path not found")
            continue
        if i == 0:
            plt.figure(figsize=(30, 30))
        plt.subplot(2, 3, valid+1)
        plt.imshow(im)
        plt.axis('off')
        if i+1 == paintings:
            plt.savefig('D:/CHBLD/sp_data/'+directory+'/' + str(sub_n) + '/sbj_'+str(sub_n) + name,
                        bbox_inches='tight')
            plt.close()
        print(valid+1)


def make_slides_subject_hm(sub_n, screen=False):
    """
    make slides containing paintings heatmaps for each subject.
    :param sub_n: subject no. (to find it in the data)
    :param screen: boolean, True if you want to use screen observation data, False if museum
    :return: nothing, it saves image slides to the given path
    to use you can change no. of "paintings" and change paths of data
    ("directory", "name" and the path passed to imread and savefig)
    """

    if screen:
        directory = "data_c"
    else:
        directory = "data_m"

    name = '_heatmap.png'
    valid = -1
    paintings = 5
    for i in range(paintings):
        try:
            im = plt.imread('D:/CHBLD/hm_data/'+directory+'/' + str(sub_n) + '/' + str(sub_n) + "_" + str(i+1) + name)
            valid += 1
        except:
            print("path not found")
            continue
        if i == 0:
            plt.figure(figsize=(30, 30))
        plt.subplot(2, 3, valid+1)
        plt.imshow(im)
        plt.axis('off')
        if i+1 == paintings:
            plt.savefig('D:/CHBLD/hm_data/'+directory+'/' + str(sub_n) + '/sbj_'+str(sub_n) + '_heatmaps.png',
                        bbox_inches='tight')
            plt.close()
        print(valid+1)
