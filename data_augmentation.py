import math
import tensorflow as tf


class RandomSpeed(tf.keras.layers.Layer):
    def __init__(self, frames=128, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        p = tf.cast(0.75 * self.frames, tf.int32)
        x_min = tf.cond(height < p, lambda: height, lambda: p)
        x_max = self.frames + 1
        x = tf.random.uniform(shape=[], minval=x_min, maxval=x_max,
                              dtype=tf.int32, seed=self.seed)
        resized_images = tf.image.resize(images, [x, width])
        # paddings = [[0, 0], [0, self.frames - x], [0, 0], [0, 0]]
        # padded_images = tf.pad(resized_images, paddings, "CONSTANT")

        if self.debug:
            tf.print("speed", x)

        # return padded_images
        return resized_images


class RandomScale(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def round_down_float_to_1_decimal(self, num):
        return tf.math.floor(num * 10.0) / 10.0

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        red_maxs = tf.reduce_max(red, axis=-1, keepdims=True)
        red_mins = tf.reduce_min(red, axis=-1, keepdims=True)
        red_mids = (red_maxs + red_mins) / 2
        red_alphas_1 = (self.min_value - red_mids) / (red_mins - red_mids)
        red_alphas_2 = (self.max_value - red_mids) / (red_maxs - red_mids)
        red_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([red_alphas_1, red_alphas_2]))

        green_maxs = tf.reduce_max(green, axis=-1, keepdims=True)
        green_mins = tf.reduce_min(green, axis=-1, keepdims=True)
        green_mids = (green_maxs + green_mins) / 2
        green_alphas_1 = (self.min_value - green_mids) / \
            (green_mins - green_mids)
        green_alphas_2 = (self.max_value - green_mids) / \
            (green_maxs - green_mids)
        green_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([green_alphas_1, green_alphas_2]))

        max_alpha = tf.reduce_min([red_alpha, green_alpha])
        alpha = tf.random.uniform(
            shape=[], minval=0.5, maxval=max_alpha, seed=self.seed)
        new_red = alpha * (red - red_mids) + red_mids
        new_green = alpha * (green - green_mids) + green_mids

        if self.debug:
            tf.print("scale", alpha)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomShift(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        left_offset = tf.reduce_min(red) - self.min_value
        right_offset = self.max_value - tf.reduce_max(red)
        red_shift = tf.random.uniform(shape=[],
                                      minval=tf.math.negative(left_offset),
                                      maxval=right_offset,
                                      seed=self.seed)

        if self.debug:
            tf.print("red shift", red_shift)

        bottom_offset = tf.reduce_min(green) - self.min_value
        top_offset = self.max_value - tf.reduce_max(green)
        green_shift = tf.random.uniform(shape=[],
                                        minval=tf.math.negative(bottom_offset),
                                        maxval=top_offset,
                                        seed=self.seed)

        new_red = tf.add(red, red_shift)
        new_green = tf.add(green, green_shift)

        if self.debug:
            tf.print("green shift", green_shift)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, factor=45.0, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_degree = tf.math.negative(factor)
        self.max_degree = factor
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        degree = tf.random.uniform(shape=[],
                                   minval=self.min_degree,
                                   maxval=self.max_degree,
                                   seed=self.seed)
        if self.debug:
            tf.print("degree", degree)

        angle = degree * math.pi / 180.0

        [red, green, blue] = tf.unstack(image, axis=-1)
        mid_value = self.min_value + (self.max_value - self.min_value) / 2
        new_red = mid_value + \
            tf.math.cos(angle) * (red - mid_value) - \
            tf.math.sin(angle) * (green - mid_value)
        new_green = mid_value + \
            tf.math.sin(angle) * (red - mid_value) + \
            tf.math.cos(angle) * (green - mid_value)
        new_red = tf.clip_by_value(new_red, self.min_value, self.max_value)
        new_green = tf.clip_by_value(new_green, self.min_value, self.max_value)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomFlip(tf.keras.layers.Layer):
    def __init__(self, mode, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        rand = tf.random.uniform(shape=[],
                                 minval=0.,
                                 maxval=1.,
                                 seed=self.seed)
        [red, green, blue] = tf.unstack(image, axis=-1)
        flip_horizontal = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'horizontal'))
        flip_vertical = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'vertical'))
        new_red = tf.cond(
            flip_horizontal, lambda: tf.add(-red, self.max_value), lambda: red)
        new_green = tf.cond(
            flip_vertical, lambda: tf.add(-green, self.max_value), lambda: green)

        if self.debug:
            tf.print("flip", rand)

        return tf.stack([new_red, new_green, blue], axis=-1)
