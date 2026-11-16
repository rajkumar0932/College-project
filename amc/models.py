"""CNN models for AMC (TensorFlow / Keras). Input: raw I/Q, shape (128, 2)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DISPLAY_NAMES = {
    "vtcnn2": "VT-CNN2 (baseline)",
    "rescnn": "Proposed ResCNN-1D",
    "rescnn_aug": "Proposed ResCNN-1D + rotation aug.",
    "feat_dt": "Features + Decision Tree",
    "feat_rf": "Features + Random Forest",
}


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def vt_cnn2(n_classes: int = 11, input_shape=(128, 2), dropout: float = 0.5) -> keras.Model:
    """VT-CNN2 from O'Shea, Corgan & Clancy, "Convolutional Radio Modulation
    Recognition Networks" (2016), the reference CNN for RML2016.10a.

    I/Q is treated as a 2 x 128 single-channel "image".
    """
    T, C = input_shape
    inp = keras.Input(shape=input_shape, name="iq")
    x = layers.Permute((2, 1))(inp)                         # (2, 128)
    x = layers.Reshape((C, T, 1))(x)                        # (2, 128, 1)
    x = layers.ZeroPadding2D((0, 2))(x)
    x = layers.Conv2D(256, (1, 3), activation="relu", name="conv1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.ZeroPadding2D((0, 2))(x)
    x = layers.Conv2D(80, (2, 3), activation="relu", name="conv2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal", name="dense1")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="VT-CNN2")


def _res_block(x, filters: int, kernel: int = 3, name: str = "res"):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = layers.Conv1D(filters, kernel, padding="same", use_bias=False, name=f"{name}_c1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_r1")(y)
    y = layers.Conv1D(filters, kernel, padding="same", use_bias=False, name=f"{name}_c2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def res_cnn1d(n_classes: int = 11, input_shape=(128, 2),
              widths=(32, 48, 64, 96), dropout: float = 0.4) -> keras.Model:
    """Proposed lightweight residual 1-D CNN.

    I and Q are the two input channels of a 1-D convolution over time, so
    every filter sees both components together (constellation information).
    Each stage halves the time axis; residual connections and BatchNorm
    make the deeper stack easy to train. Global average and max pooling
    replace the large Flatten+Dense of VT-CNN2, which removes about 95% of
    the parameters and keeps inference fast (proposal Section 4.4).
    """
    inp = keras.Input(shape=input_shape, name="iq")
    x = layers.Conv1D(widths[0], 7, padding="same", use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    for i, w in enumerate(widths):
        x = _res_block(x, w, name=f"stage{i + 1}")
        x = layers.MaxPooling1D(2, name=f"pool{i + 1}")(x)       # 128 -> 64 -> 32 -> 16 -> 8
    x = layers.Concatenate(name="pool_concat")([
        layers.GlobalAveragePooling1D(name="gap")(x),
        layers.GlobalMaxPooling1D(name="gmp")(x),
    ])
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="ResCNN-1D")


BUILDERS = {"vtcnn2": vt_cnn2, "rescnn": res_cnn1d}


def build_model(name: str, n_classes: int = 11) -> keras.Model:
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BUILDERS)}")
    return BUILDERS[name](n_classes=n_classes)


def random_rotation(x, y):
    """Augmentation: rotate each I/Q example by a random phase.

    The channel already applies an unknown phase, so rotation keeps the label.
    """
    theta = tf.random.uniform((tf.shape(x)[0], 1), 0.0, 2.0 * 3.141592653589793)
    c, s = tf.cos(theta), tf.sin(theta)
    i, q = x[..., 0], x[..., 1]
    return tf.stack([i * c - q * s, i * s + q * c], axis=-1), y
