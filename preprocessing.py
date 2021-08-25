import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from config import *


def resize_only(data):
    # bbox = data['objects']['bbox']
    # bbox = data['bbox']
    image = data['image']
    label = data['label']
    image = tf.image.resize(images=image, size=NEW_SIZE)
    return image, label


def crop_by_bbox(data):
    # bbox = data['objects']['bbox'][0]
    bbox = data['bbox']
    image = data['image']
    label = data['label']
    image = tf.image.crop_and_resize(image=tf.expand_dims(image, axis=0), boxes=[bbox], box_indices=[0],
                                     crop_size=NEW_SIZE)
    image = tf.squeeze(image)
    image = tf.image.resize(images=image, size=NEW_SIZE)
    return image, label


def augment(bbox_image_label, seed):
    bbox, image, label = bbox_image_label
    # image = tf.image.crop_and_resize(image=tf.expand_dims(image, axis=0), boxes=[bbox], box_indices=[0],
    #                                  crop_size=NEW_SIZE)
    # image = tf.squeeze(image)
    image = tf.image.resize(images=image, size=NEW_SIZE)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32))
    # image=tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)(image, training=True)
    # image = tf.squeeze(image)
    # image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    seed1 = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed1)
    seed2 = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_contrast(image, lower=0.5, upper=1.5, seed=seed2)
    seed3 = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed3)
    # image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)(image)
    # image = tf.keras.layers.GaussianNoise(10)(image, training=True)
    return bbox, image, label


def test_image(dataset):
    for b, c in dataset.take(1):
        # print(a)
        # print(b)
        # print(c)
        for i in range(30):
            # print(a[i])
            # print(b[i])
            plt.imshow(b[i] / 255)
            plt.show()


def load_data_and_preprocessing(crop=False):
    [train_ds, val_ds, test_ds], ds_info = tfds.load('caltech_birds2011', split=["train[:80%]", "train[80%:]", "test"],
                                                     as_supervised=False,
                                                     shuffle_files=True,
                                                     with_info=True, data_dir=DATASETS_DIR)
    num_classes = ds_info.features['label'].num_classes

    preprocess = crop_by_bbox if crop else resize_only
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.map(preprocess).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)
    # print(train_ds)
    # print(test_ds)
    # test_image(test_ds)
    return train_ds, val_ds, test_ds, num_classes


if __name__ == "__main__":
    train_ds, val_ds, test_ds, num_classes = load_data_and_preprocessing(True)
