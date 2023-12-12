"""
instance_transformations.py

Applies per-instance transformations to a batch of data.

Much of this implementation is based on the TF implementation of
SimCLR v2 at https://github.com/google-research/simclr

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import functools
import tensorflow.compat.v2 as tf


def random_apply(func, p, x):
    """
    Randomly apply function func to x with probability p.
    """
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


def random_brightness(image, max_delta):
    """
    Randomly distorts the brightness of an image.

    Args:
        image: the input image tensor.
        max_delta: the maximum magnitude difference from 1.0 that your
            multiplicative brightness adjustment factor will be.
    Returns:
        The distorted image tensor
    """
    factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                               1.0 + max_delta)
    image = image * factor
    return image


def to_grayscale(image):
    """
    Converts an image from RBG to grayscale. Maintains the same dimensions.
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


def color_jitter(image, strength):
    """
    Distorts the color of the image by distorting the brightness, contrast,
    saturation, and hue. Applies distortions in random order.

    Args:
        image: The input image tensor.
        strength: the floating number for the strength of the color augmentation.
    Returns:
        The distorted image tensor.
    """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength

    with tf.name_scope('distort_color'):

        def apply_transform(i, x):
            """
            Apply the i-th transformation.
            """

            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness)

            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(x,
                                                    lower=1 - contrast,
                                                    upper=1 + contrast)

            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(x,
                                                      lower=1 - saturation,
                                                      upper=1 + saturation)

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)

            x = tf.cond(
                tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0., 1.)
        return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    """
    Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: Tensor of image data.
        bbox: Tensor of bounding boxes arranged `[1, num_boxes, coords]`
            where each coordinate is [0, 1) and the coordinates are arranged
            as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding
            box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
    Returns:
        Cropped image tensor
    """
    with tf.name_scope('distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box
        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                              target_height, target_width)
        return image


def random_crop_with_resize(image, height, width, p=1.0):
    """
    Randomly crop and resize an image

    Args:
        image: Tensor representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        p: Probability of applying this transformation.
    Returns:
        The cropped and resized image tensor
    """

    def _transform(image):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
        aspect_ratio = width / height
        image = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.08, 1.0),
            max_attempts=100)
        return tf.image.resize([image], [height, width],
                               method=tf.image.ResizeMethod.BICUBIC)[0]

    return random_apply(_transform, p=p, x=image)


def random_color_jitter_plus_grayscale(image, p=1.0, strength=0.9):
    """
    Apply a random color jitter followed by randomly converting to grayscale.

    Args:
        image: Tensor of image data.
        p: Probability of applying either (randomly occurring) transformation
        strength: Scales the strength of all color transformations.
    Returns:
        The distorted image tensor.
    """

    def _transform(image):
        color_jitter_t = functools.partial(color_jitter, strength=strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)

    return random_apply(_transform, p=p, x=image)


def transform_image(image,
                    height=32,
                    width=32,
                    color_distort=True,
                    crop=True,
                    flip=True):
    """
    Transforms the given image randomly.

    Args:
        image: Tensor representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        color_distort: Whether to allow color distortions.
        crop: Whether to allow cropping the image.
        flip: Whether or not to allow flipping left and right of an image.
    Returns:
        A transformed image tensor
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter_plus_grayscale(image)

    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def get_rnd_trans(inp):
    """
    Applys transformations to a batch of images. Returns the transformed batch.
    """
    return tf.map_fn(transform_image, inp)
