# Imports
import heatmap as hm
import matplotlib.pyplot as plt
import scan_path as sp
import make_slides as ms


def main():
    """
    Generates heatmaps and scan paths,
    to use you have to change the path in here and in the functions
    """
    subjects = 22
    paintings = 5
    # make heat maps, scan paths of the first seconds and scan paths of the whole rec (30 sec)
    # and scan paths for 15 sec
    for subj in range(1, subjects+1):
        for painting in range(1, paintings+1):
            painting_im = plt.imread('D:/CHBLD/sp_data/data_c/painting_' + str(painting) + '.jpg')

            hm.generate_heatmap(painting_im, subj, painting)
            hm.generate_heatmap(painting_im, subj, painting, True)

            sp.generate_scan_path(painting_im, subj, painting, False)
            sp.generate_scan_path(painting_im, subj, painting, True, 15)
            sp.generate_scan_path(painting_im, subj, painting, False, screen=True)
            sp.generate_scan_path(painting_im, subj, painting, True, 15, True)

    # make slides for each subject (slide containing the 5 paintings for each subject)
    for subj in range(1, subjects+1):
        ms.make_slides_subject_hm(subj)
        ms.make_slides_subject_sp(subj)
        ms.make_slides_subject_sp(subj, True, 15)
        ms.make_slides_subject_hm(subj, screen=True)
        ms.make_slides_subject_sp(subj, screen=True)
        ms.make_slides_subject_sp(subj, True, 15, screen=True)

    # make slides for each painting (slides containing all the subjects for each painting)
    for painting in range(1, paintings+1):
        ms.make_slides_hm(painting)
        ms.make_slides_sp(painting)
        ms.make_slides_sp(painting, True, 15)
        ms.make_slides_hm(painting, screen=True)
        ms.make_slides_sp(painting, screen=True)
        ms.make_slides_sp(painting, True, 15, screen=True)


if __name__ == '__main__':
    main()
