import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def conv2d_block(input_tensor, n_filters, kernel_size=3, batch_norm=True):
    """Function to add 2 convolution layers with the parameters passed to it

    :param input_tensor: input of the first convolution layer.
    :type input_tensor: tensor
    :param n_filters: filter size of first convolution layer.
    :type n_filters: int
    :param kernel_size: size of the kernel filter. Defaults to 3.
    :type kernel_size: int
    :param batch_norm: if set, activate batch normalization. Defaults to True.
    :type batch_norm: boolean
    ...

    :return: output of last layer in this block
    :rtype: tensor

    """
    # first layer
    x = layers.Conv2D(filters=n_filters,
                      kernel_size=(kernel_size, kernel_size),
                      padding='same')(input_tensor)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # second layer
    x = layers.Conv2D(filters=n_filters,
                      kernel_size=(kernel_size, kernel_size),
                      kernel_initializer='he_normal',
                      padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def get_u_net(input_img, num_classes, n_filters=16, batch_norm=True):
    """ U-Net model

    :param input_img: input of U-Net architecture.
    :type input_img: tensor
    :param num_classes: number of output classes.
    :type num_classes: int
    :param n_filters: filter size of the first convolution layer.
        Defaults to 16.
    :type n_filters: int
    :param batch_norm: if set, batch normalization is enabled.
        Defaults to True.
    :type batch_norm: boolean
    ...

    :return: model architecture
    :rtype: tensors

    """
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3,
                      batch_norm=batch_norm)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batch_norm=batch_norm)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batch_norm=batch_norm)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batch_norm=batch_norm)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3,
                      batch_norm=batch_norm)

    # Expansive Path
    u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3),
                                strides=(2, 2),
                                padding='same')(c5)
    u6 = layers.concatenate([u6, c4], axis=-1)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batch_norm=batch_norm)

    u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3),
                                strides=(2, 2),
                                padding='same')(c6)
    u7 = layers.concatenate([u7, c3], axis=-1)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batch_norm=batch_norm)

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3),
                                strides=(2, 2),
                                padding='same')(c7)
    u8 = layers.concatenate([u8, c2], axis=-1)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batch_norm=batch_norm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3),
                                strides=(2, 2),
                                padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=-1)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batch_norm=batch_norm)

    output = layers.Conv2D(num_classes, (1, 1),
                           padding='same',
                           activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[output])

    return model


def main():
    """Displays model summary

    """
    input_img = Input(shape=(256, 256, 3), name='img')
    model = get_u_net(input_img, num_classes=19)
    model.summary()


if __name__ == '__main__':
    main()
