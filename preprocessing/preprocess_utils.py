import os
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           '../results/visualization')


def make_folder(path):
    """Check if the folder exists,
    if it doesn't exist create one in the given path.

    :param path: path where the folder needs to be created.
    :type path: string

    """
    if not os.path.exists(os.path.join(path)):
        print('[INFO] Creating new folder...')

        os.makedirs(os.path.join(path))


def display(display_list, titles, result):
    """ Store the row of given images in the given filename.

    :param display_list: list of images to display.
    :type display_list: list of str
    :param titles: list of titles of the images to display.
    :type titles: list of str
    :param result: filename in which the plot is saved in results folder.
    :type result: str

    """
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    make_folder(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, '{}.png'.format(result)))
    print('[ACTION] See results/visualization/{}.png'.format(result))
