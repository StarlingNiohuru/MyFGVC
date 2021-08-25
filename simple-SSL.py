import matplotlib.pyplot as plt
from config import *
from preprocessing import load_data_and_preprocessing

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(num_classes):
    input_tensor = tf.keras.layers.Input(INPUT_TENSOR_SHAPE)
    x = tf.cast(input_tensor, tf.float32)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip()(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)
    x = tf.keras.layers.GaussianNoise(10)(x)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    pre_train_model = tf.keras.applications.VGG16(weights="imagenet", input_tensor=x, include_top=False)
    pre_train_model.trainable = False
    # x = tf.keras.layers.Flatten()(pre_train_model.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(pre_train_model.output)
    # x = tf.keras.layers.Dense(4096, activation='relu')(x)
    # x = tf.keras.layers.Dropout(DROP_OUT)(x)
    # x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROP_OUT)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    # sgd = tf.optimizers.SGD(learning_rate=1e-3, decay=0.0, momentum=0.9, nesterov=False)
    adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = tf.keras.Model(inputs=input_tensor, outputs=predictions)
    model.summary()
    model.compile(optimizer=adam,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


def load_model():
    return tf.keras.models.load_model(SSL_MODEL_PATH)


def fine_tune(model):
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    sgd = tf.optimizers.SGD(learning_rate=1e-5, decay=1e-9, momentum=0.9, nesterov=False)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=sgd,
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
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=SSL_MODEL_PATH, save_best_only=True,
                                                                   verbose=1,
                                                                   monitor='accuracy', mode='max')
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
    # early_stopping_callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
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
    train_ds, val_ds, test_ds, num_classes = load_data_and_preprocessing(True)
    # build_model(num_classes)
    # history = train_model(train_ds, val_ds, test_ds, num_classes)
    history = train_model(train_ds, val_ds, test_ds, num_classes, True)
    show_history_graph(history)
