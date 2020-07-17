import os
import numpy as np
import cv2
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Input
from model import u_net
from preprocessing.preprocess_utils import display
from experiments import lip_hair_color


def make_confusion_matrix(cf, categories,
                          group_names=None,
                          count=True,
                          percent=True,
                          color_bar=True,
                          xy_ticks=True,
                          xy_plot_labels=True,
                          sum_stats=True,
                          fig_size=None,
                          c_map='Blues',
                          title=None):
    """ Code to generate text within each box and beautify confusion matrix.

    :param cf: Confusion matrix.
    :type cf: numpy array
    :param categories: array of classes.
    :type categories: numpy array
    :param group_names: classes in the project.
    :type group_names: numpy array
    :param count: whether to display the count of each class.
    :type count: boolean
    :param percent: whether to display percentage for each class.
    :type percent: boolean
    :param color_bar: whether to display color bar for the heat map.
    :type color_bar: boolean
    :param xy_ticks: whether to display xy labels.
    :type xy_ticks: boolean
    :param xy_plot_labels: whether to display xy title.
    :type xy_plot_labels: boolean
    :param sum_stats: whether to display overall accuracy.
    :type sum_stats: boolean
    :param fig_size: size of the plot.
    :type fig_size: tuple
    :param c_map: color scheme to use.
    :type c_map: str
    :param title: Title of the plot.
    :type title: str

    """
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        row_size = np.size(cf, 0)
        col_size = np.size(cf, 1)
        group_percentages = []
        for i in range(row_size):
            for j in range(col_size):
                group_percentages.append(cf[i][j] / cf[i].sum())
        group_percentages = ["{0:.2%}".format(value)
                             for value in group_percentages]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip()
                  for v1, v2, v3 in zip(group_labels,
                                        group_counts,
                                        group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        stats_text = "\n\nAccuracy={0:0.2%}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if fig_size is None:
        # Get default figure size if not set
        fig_size = plt.rcParams.get('figure.figsize')

    if not xy_ticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEAT MAP VISUALIZATION
    plt.figure(figsize=fig_size)
    sns.heatmap(cf, annot=box_labels, fmt="",
                cmap=c_map, cbar=color_bar,
                xticklabels=categories,
                yticklabels=categories)

    if xy_plot_labels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def plot_confusion_matrix(predictions, masks, path):
    """ Visualize confusion matrix.

    :param predictions: predicted output of the model.
    :type predictions: array
    :param masks: true masks of the images.
    :type masks: array
    :param path: directory to store the output
    :type path: str

    """
    print('[INFO] Plotting confusion matrix...')
    corr = confusion_matrix(masks.ravel(), predictions.ravel())
    make_confusion_matrix(corr,
                          categories=['bg', 'skin', 'nose', 'eye_g', 'l_eye',
                                      'r_eye', 'l_brow', 'r_brow', 'l_ear',
                                      'r_ear', 'mouth', 'u_lip',
                                      'l_lip', 'hair', 'hat', 'ear_r',
                                      'neck_l', 'neck', 'cloth'],
                          count=True,
                          percent=False,
                          color_bar=False,
                          xy_ticks=True,
                          xy_plot_labels=True,
                          sum_stats=True,
                          fig_size=(20, 18),
                          c_map='coolwarm',
                          title='Confusion matrix')

    # error correction - cropped heat map
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    print('[ACTION] See results/visualization/confusion_matrix.png')


def plot_mask(prediction, mask, norm_image):
    """ PLot segmentation mask for the given image.

    :param prediction: predicted output of the model.
    :type prediction: array
    :param mask: true masks of the images.
    :type mask: array
    :param norm_image: original image.
    :type norm_image: array

    """
    image = (norm_image * 255.).astype(np.uint8)
    im_base = np.zeros((256, 256, 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        im_base[prediction == idx] = color
    cv2.addWeighted(im_base, 0.8, image, 1, 0, im_base)
    display([image, mask, im_base],
            ['Original image', 'True mask', 'Predicted mask'],
            'predict')


def test(image, masks, action, color='red'):
    """ Used to plot either confusion matrix or predicted mask or apply makeup.

    :param image: original image.
    :type image: bytearray
    :param masks: true segmentation masks.
    :type masks: array
    :param action: user input specifying confusion matrix/mask
    prediction/applying makeup.
    :type action: str
    :param color: if action is applying makeup, then color to apply.
        Defaults to red.
    :type color: str

    """
    input_img = Input(shape=(256, 256, 3), name='img')
    model = u_net.get_u_net(input_img, num_classes=19)
    model.load_weights(os.path.join(MODEL_DIR, 'u_net.h5'))

    print('[INFO] Predicting ...')
    predictions = model.predict(image)
    predictions = np.argmax(predictions, axis=-1)

    table = {
        'hair': 13,
        'upper_lip': 11,
        'lower_lip': 12
    }

    colors = {
        'red': [212, 34, 34],
        'purple': [128, 51, 125],
        'pink': [247, 32, 125]
    }

    # Redirect to the function of specified action.
    if action == 'confusion_matrix':
        print('[INFO] Plotting confusion matrix ...')
        plot_confusion_matrix(predictions, masks, VISUALIZATION_DIR)

    elif action == 'mask':
        print('[INFO] Plotting segmentation mask ...')
        plot_mask(predictions[sample], masks[sample], image[sample])

    elif action == 'hair_color':
        print('[INFO] Applying hair color ...')
        parts = [table['hair']]
        changed = lip_hair_color.color_change(image[sample],
                                              predictions[sample],
                                              parts, colors[color])
        display([image[sample], changed], 'hair')

    elif action == "lip_color":
        print('[INFO] Applying lip color ...')
        parts = [table['upper_lip'], table['lower_lip']]
        changed = lip_hair_color.color_change(image[sample],
                                              predictions[sample],
                                              parts, colors[color])
        display([image[sample], changed], 'lip')


def main():
    """ Define user arguments.

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--visualize", type=str, required=True,
                    choices=("confusion_matrix", "mask",
                             "hair_color", "lip_color"),
                    help="type of model")
    ap.add_argument("-c", "--color", type=str,
                    choices=("red", "pink", "purple"),
                    help="color to apply")
    args = vars(ap.parse_args())

    # print('[INFO] Getting test data...')
    # test_data = get_test()
    # imgs = []
    # masks = []
    # for img, label in test_data:
    #     for i in img:
    #         i = np.array(i, dtype='float32')
    #         imgs.append(i)
    #     for j in label:
    #         j = np.array(j, dtype='float32')
    #         masks.append(j)
    # images = np.array(imgs)
    # masks = np.array(masks)
    # np.save('data/test_images.npy', images)
    # np.save('data/test_mask.npy', masks)

    # Load test images
    images = np.load('data/test_images.npy')
    masks = np.load('data/test_mask.npy')
    test(images, masks, args["visualize"], args["color"])


if __name__ == '__main__':
    VISUALIZATION_DIR = 'results/visualization/'
    MODEL_DIR = 'results/models/'
    color_list = [[0, 0, 0], [204, 0, 0], [255, 140, 26],
                  [204, 204, 0], [51, 51, 255], [204, 0, 204],
                  [0, 255, 255], [255, 204, 204], [102, 51, 0],
                  [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153],
                  [0, 204, 204], [0, 51, 0], [255, 153, 51],
                  [0, 204, 0]]
    sample = 4
    main()
