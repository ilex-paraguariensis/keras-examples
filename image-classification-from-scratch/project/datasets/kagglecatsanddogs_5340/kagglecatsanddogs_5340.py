import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def kagglecatsanddogs_5340(batch_size: int = 32):
    image_size = (180, 180)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )
    return augmented_train_ds.prefetch(buffer_size=32), val_ds.prefetch(
        buffer_size=32
    )
