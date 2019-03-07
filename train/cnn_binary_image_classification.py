# CNN Extraction with data Augmentation and tuneup
# ~7 hours on Dell latop with 92% validation accuracy
# data in two folders: notover and over
# Tools:
#   Keras 2.2.4
#   TensorFLow latest
#   Python 3.6
#   Note: install missing Python modules as required..,

# Imports
import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# Set variables
base_dir = r'C:\Users\abayk_000\Documents\A Knowledge base\AWS\Deep Learning\Cats vs Dogs\data2'
training_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validate')
testing_dir = os.path.join(base_dir, 'test')

# Instantiate VGG16 convolutional base
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 224, 3))

# Add densely connected classifier on top of the pretrained base
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Freeze the base (the most of them)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Need below for debugging pupose only to look at the sructure of the base
# model.summary()

# Train end to end with base frozen
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        training_dir,
        # All images will be resized to 150x150
        target_size=(150, 224),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

#Print classes
print("Classes: ", train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 224),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=80,
      epochs=27,
# he had 100 epochs
      validation_data=validation_generator,
      validation_steps=10)
#      ,verbose=2)

# Save the model
model.save('overload3_1.h5')

# Save the weights
model.save_weights('overload3_1_weights.h5')

# Save the model architecture
with open('overload3_1_architecture.json', 'w') as f:
    f.write(model.to_json())

# Gather plot data
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Plot the results
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Test
test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(150, 224),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=5)
print('test acc:', test_acc)
