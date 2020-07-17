import os
import cv2
import numpy as np
import pandas as pd
import h5py
from preprocessing import preprocess_utils


def combine_mask():
    """ Combines all the individual label masks in folder
    `CeleAMaskHQ-mask` to generate a single masked image
    for the corresponding image in folder 'CelebA-HQ-img`.

    :return: Creates a folder `CelebAMaskHQ-mask` with mask
    images of the corresponding images in 'CelebA-HQ-img'.
    :rtype: folder

    """
    folder_base = os.path.join(ROOT_DIR, 'CelebAMask-HQ-mask-anno/')
    folder_save = os.path.join(ROOT_DIR, 'CelebAMaskHQ-mask/')
    preprocess_utils.make_folder(folder_save)

    print('[INFO] Generating masks...')
    for k in range(img_num):
        # finds the folder in which the individual masks are present
        folder_num = int(k / 2000)
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, str(folder_num),
                                    str(k).rjust(5, '0') + '_'
                                    + label + '.png')
            if os.path.exists(filename):
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)

        im_base = cv2.resize(im_base, (256, 256), cv2.INTER_AREA)
        filename_save = os.path.join(folder_save, str(k) + '.png')
        cv2.imwrite(filename_save, im_base)


def split_data(train_count, valid_count, test_count):
    """Split the CelebA-HQ data set into train, validation and
    test data set as mentioned in the 'list_eval_partition.txt'
    file of CelebA data set. Store this file in hdf5 format.

    :return: Three hdf5 file having images and masks of model,
    validation and test data sets.
    :rtype: hdf5 file

    """
    print('[INFO] Splitting data into train, validation and test...')

    # Get CelebA-HQ to CelebA mapping into a data frame
    relation_path = os.path.join(ROOT_DIR, 'CelebA-HQ-to-CelebA-mapping.txt')
    relation = pd.read_csv(relation_path, delim_whitespace=True,
                           na_filter=False)

    # Get partitioning sequence into a data frame
    split_path = os.path.join(ROOT_DIR, 'list_eval_partition.txt')
    splitting = pd.read_csv(split_path, delim_whitespace=True,
                            header=None, names=['img', 'group'],
                            na_filter=False)

    file = h5py.File(os.path.join(DATA_DIR, 'raw.h5'), "w")

    # Map CelebA-HQ data set with CelebA and split accordingly.
    for t in range(img_num):
        celebA = relation.iloc[t]['orig_idx']
        group_id = splitting.iloc[celebA]['group']
        X_filename = os.path.join(ROOT_DIR, 'CelebA-HQ-img/' + str(t) + '.jpg')
        y_filename = os.path.join(ROOT_DIR, 'CelebAMaskHQ-mask/'
                                  + str(t) + '.png')
        img = cv2.imread(X_filename)
        img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
        mask = cv2.imread(y_filename, cv2.IMREAD_UNCHANGED)
        if group_id == 0:
            file.create_dataset("train/images/{}".format(train_count),
                                img.shape, h5py.h5t.STD_U8BE, data=img)
            file.create_dataset("train/meta/{}".format(train_count),
                                mask.shape, h5py.h5t.STD_U8BE, data=mask)
            train_count += 1
        elif group_id == 1:
            file.create_dataset("valid/images/{}".format(valid_count),
                                img.shape, h5py.h5t.STD_U8BE, data=img)
            file.create_dataset("valid/meta/{}".format(valid_count),
                                mask.shape, h5py.h5t.STD_U8BE, data=mask)
            valid_count += 1
        elif group_id == 2:
            file.create_dataset("test/images/{}".format(test_count),
                                img.shape, h5py.h5t.STD_U8BE, data=img)
            file.create_dataset("test/meta/{}".format(test_count),
                                mask.shape, h5py.h5t.STD_U8BE, data=mask)
            test_count += 1

    file.close()
    # print number of train, valid and test images
    print('Image shape = {}'.format(str(img.shape)))
    print('Mask shape = {}'.format(str(mask.shape)))
    print('Train images = {}'.format(train_count))
    print('Validation images = {}'.format(valid_count))
    print('Test images = {}'.format(test_count))

    return train_count, valid_count, test_count


def main():
    """Create masks of each image.
    Split the data set into model, validation and test data.

    """
    combine_mask()
    train = valid = test = 0
    train, valid, test = split_data(train, valid, test)


if __name__ == "__main__":
    label_list = ['skin', 'nose', 'eye_g', 'l_eye',
                  'r_eye', 'l_brow', 'r_brow', 'l_ear',
                  'r_ear', 'mouth', 'u_lip',
                  'l_lip', 'hair', 'hat', 'ear_r',
                  'neck_l', 'neck', 'cloth']
    ROOT_DIR = '/media/pavan/datasets/face_segmentation/CelebAMask-HQ/'
    DATA_DIR = '../data/'
    img_num = 30000
    main()
