import h5py
import os
import cv2
import tensorflow as tf
from preprocessing.preprocess_utils import display

tf.compat.v1.enable_eager_execution()
AUTO_TUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
BATCH_SIZE = 32


def normalize(image, label):
    """ Normalize the image to range (0,1).

    :param image: image to normalize.
    :type image: array.
    :param label: mask of the corresponding image.
    :type label: array.
    ...

    :return: normalized image and its corresponding mask.
    :rtype: array.

    """
    image = image / 255.0

    return image, label


def augment(image, label):
    """ Augment the given image.

    :param image: image to augment.
    :type image: array
    :param label: mask of the corresponding image.
    :type label: array
    ...

    :return: augmented image and its corresponding mask
    :rtype: array

    """
    image, label = normalize(image, label)
    # Random brightness
    image = tf.image.random_brightness(image,
                                       max_delta=0.5)

    return image, label


def get_train_data():
    """ Get train data from h5py file and feed it into a tensorflow dataset

    :yield: image and its corresponding mask
    :ytype: array

    """
    file = h5py.File(os.path.join(DATA_DIR, 'raw.h5'), 'r')
    folder = file['train/images/']
    train_count = len(list(folder.keys()))
    for i in range(train_count):
        train_img, train_mask = file['train/images/{}'.format(i)][()], \
                                file['train/meta/{}'.format(i)][()]
        yield (cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB), train_mask)


def get_valid_data():
    """ Get validation data from h5py file and feed it into a tensorflow dataset

    :yield: image and its corresponding mask
    :ytype: array

    """
    file = h5py.File(os.path.join(DATA_DIR, 'raw.h5'), 'r')
    folder = file['valid/images/']
    for i in range(len(list(folder.keys()))):
        train_img, train_mask = file['valid/images/{}'.format(i)][()], \
                                file['valid/meta/{}'.format(i)][()]
        yield (cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB), train_mask)


def get_test_data():
    """ Get test data from h5py file and feed it into a tensorflow dataset

    :yield: image and its corresponding mask
    :ytype: array

    """
    file = h5py.File(os.path.join(DATA_DIR, 'raw.h5'), 'r')
    folder = file['test/images/']
    for i in range(len(list(folder.keys()))):
        train_img, train_mask = file['test/images/{}'.format(i)][()], \
                                file['test/meta/{}'.format(i)][()]
        yield (cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB), train_mask)


def get_data():
    """ Create train and validation dataset of given batch size

    :return: train and validation data
    :rtype: dataset

    """
    train_data = (tf.data.Dataset.from_generator(
        get_train_data,
        (tf.float32, tf.int32),
        (tf.TensorShape([256, 256, 3]), tf.TensorShape([256, 256])))
                  .map(augment, num_parallel_calls=AUTO_TUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(AUTO_TUNE))

    valid_data = (tf.data.Dataset.from_generator(
        get_valid_data,
        (tf.float32, tf.int32),
        (tf.TensorShape([256, 256, 3]), tf.TensorShape([256, 256])))
                  .map(normalize, num_parallel_calls=AUTO_TUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(AUTO_TUNE))

    return train_data, valid_data


def get_test():
    """ Create test dataset of given batch size

     :return: test data
     :rtype: dataset

     """
    test_data = (tf.data.Dataset.from_generator(
        get_test_data,
        (tf.float32, tf.int32),
        (tf.TensorShape([256, 256, 3]), tf.TensorShape([256, 256])))
                 .map(normalize, num_parallel_calls=AUTO_TUNE)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTO_TUNE)
                 )

    return test_data


def main():
    """ Display sample images of train, validation and test data

    """
    train_data, valid_data = get_data()
    test_data = get_test()

    # Extract one image from dataset and display
    for img, label in train_data.take(1):
        sample_image, sample_mask = img[31], label[31]
    display([sample_image, sample_mask], ['Image', 'Mask'], 'train_sample')

    for img, label in valid_data.take(1):
        sample_image, sample_mask = img[31], label[31]
    display([sample_image, sample_mask], ['Image', 'Mask'], 'valid_sample')

    for img, label in test_data.take(1):

        sample_image, sample_mask = img[31], label[31]
    display([sample_image, sample_mask], ['Image', 'Mask'], 'test_sample')


if __name__ == '__main__':
    main()
