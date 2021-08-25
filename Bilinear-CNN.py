import tensorflow as tf
from config import *
from preprocessing import load_data_and_preprocessing
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def outer_product(x):
    return tf.keras.backend.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def signed_sqrt(x):
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-9)


def l2_norm(x, axis=-1):
    return tf.keras.backend.l2_normalize(x, axis=axis)


def build_model(num_classes):
    input_tensor = tf.keras.layers.Input(INPUT_TENSOR_SHAPE)
    x = tf.cast(input_tensor, tf.float32)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip()(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)
    x = tf.keras.layers.GaussianNoise(10)(x)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    pre_train_model1 = tf.keras.applications.VGG16(
        input_tensor=x,
        include_top=False,
        weights='imagenet')
    # pre_train_model.summary()
    pre_train_model1.trainable = False
    for layer in pre_train_model1.layers:
        layer._name = layer.name + '_1'

    pre_train_model2 = tf.keras.applications.VGG16(
        input_tensor=x,
        include_top=False,
        weights='imagenet')
    pre_train_model2.trainable = False
    for layer in pre_train_model2.layers:
        layer._name = layer.name + '_2'

    # Extract features form detector
    model_detector = pre_train_model1
    output_detector = model_detector.layers[-2].output
    shape_detector = model_detector.layers[-2].output_shape

    # Extract features from extractor
    model_extractor = pre_train_model2
    output_extractor = model_extractor.layers[-2].output
    shape_extractor = model_extractor.layers[-2].output_shape
    # Reshape tensor to (batch_size, total_pixels, filter_size)
    output_detector = tf.keras.layers.Reshape(
        [shape_detector[1] * shape_detector[2], shape_detector[-1]])(output_detector)
    output_extractor = tf.keras.layers.Reshape(
        [shape_extractor[1] * shape_extractor[2], shape_extractor[-1]])(output_extractor)
    # Outer-products
    x = tf.keras.layers.Lambda(outer_product, name='outer_product')([output_detector, output_extractor])
    # Reshape tensor to (batch_size, filter_size_detector*filter_size_extractor)
    x = tf.keras.layers.Reshape([shape_detector[-1] * shape_extractor[-1]])(x)
    # Signed square-root
    x = tf.keras.layers.Lambda(signed_sqrt, name='signed_sqrt')(x)
    # L2 normalization
    x = tf.keras.layers.Lambda(l2_norm, name='l2_norm')(x)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0))(x)
    # model_bilinear = tf.keras.models.Model(inputs=[tensor_input], outputs=[tensor_prediction])
    opt_sgd = tf.keras.optimizers.SGD(learning_rate=1, decay=0.0, momentum=0.9, nesterov=False)
    model = tf.keras.Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=opt_sgd, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


def load_model():
    return tf.keras.models.load_model(BCNN_MODEL_PATH)


def fine_tune(model):
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    opt_sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-9, momentum=0.9, nesterov=False)
    model.compile(optimizer=opt_sgd,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])


def train_model(train_ds, val_ds, test_ds, num_classes, is_finetune=False):
    try:
        model = load_model()
    except:
        model = build_model(num_classes)
    if is_finetune:
        fine_tune(model)
    model.summary()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=BCNN_MODEL_PATH, save_best_only=True,
                                                                   verbose=1,
                                                                   monitor='accuracy', mode='max')
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS, shuffle=True,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
    model.evaluate(test_ds, verbose=1)
    return history


def show_history_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    train_ds, val_ds, test_ds, num_classes = load_data_and_preprocessing()
    build_model(num_classes)
    # history = train_model(train_ds, val_ds, test_ds, num_classes)
    history = train_model(train_ds, val_ds, test_ds, num_classes, True)
    show_history_graph(history)
