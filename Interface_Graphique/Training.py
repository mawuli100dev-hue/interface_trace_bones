import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input, ResNet50

# ---------------------- Argument parser ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=True, help='Chemin vers le dossier d\'entraînement')
parser.add_argument('--validation_dir', required=True, help='Chemin vers le dossier de validation')
parser.add_argument('--save_path', required=True, help='Chemin où sauvegarder le modèle (.h5)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size par défaut (non utilisé ici)')
parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques')
parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Utiliser CPU ou GPU')
args = parser.parse_args()

train_dir = args.train_dir
validation_dir = args.validation_dir
save_path = args.save_path
epochs = args.epochs
device = args.device

# ---------------------- Device configuration ----------------------
if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TensorFlow version:", tf.__version__)
print("Dispositifs GPU disponibles :", tf.config.list_physical_devices('GPU'))

# ---------------------- Auto compute dataset stats ----------------------
def count_total_images(directory):
    total = 0
    for subdir in os.listdir(directory):
        subpath = os.path.join(directory, subdir)
        if os.path.isdir(subpath):
            total += len([
                f for f in os.listdir(subpath)
                if os.path.isfile(os.path.join(subpath, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
    return total

def count_classes(directory):
    return len([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])

train_count = count_total_images(train_dir)
val_count = count_total_images(validation_dir)
num_classes = count_classes(train_dir)

print(f"Images entraînement : {train_count}, validation : {val_count}")
print(f"Nombre de classes détectées : {num_classes}")

# ---------------------- Data augmentation ----------------------
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(250, 200),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(250, 200),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# Full load to numpy arrays for training
datagenTrain = train_datagen.flow_from_directory(
    train_dir,
    target_size=(250, 200),
    batch_size=train_count,
    class_mode='categorical'
)

datagenTest = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(250, 200),
    batch_size=val_count,
    class_mode='categorical',
    shuffle=False
)

x_train, y_train = next(datagenTrain)
x_test, y_test = next(datagenTest)

print(x_train.shape, 'train samples')
print(x_test.shape, 'validation samples')

# ---------------------- Model definition ----------------------
def define_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(250, 200, 3))
    for layer in base_model.layers:
        layer.trainable = False

    flat = Flatten()(base_model.output)
    dense = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=base_model.input, outputs=output)

    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- Training ----------------------
model = define_model()
device_context = tf.device('/GPU:0' if device == 'gpu' else '/CPU:0')
with device_context:
    history = model.fit(
        train_generator,
        steps_per_epoch=30,
        validation_data=val_generator,
        validation_steps=15,
        epochs=args.epochs,
        verbose=1
    )
# ---------------------- Save ----------------------
model.save(save_path)
print(f"Model Saved  : {save_path}")
