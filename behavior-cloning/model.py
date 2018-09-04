import csv
import cv2
import numpy as np
import sklearn
import os
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Making the model to train using a speficic GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#reading the csv measurements
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

print()
print('Prepared Samples')
print()

# Preparing the train and test datasets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print()
print('Prepared train, test split')
print()

# Creating the generator
def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    """
    return image / 127.5 - 1.

def generator(samples, batch_size=32, correction = 0.2):
    num_samples = len(samples)
    processedlines = 0

    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image = center_image[40:-15,:]
                center_image = cv2.resize(center_image, (200, 66), interpolation=cv2.INTER_AREA)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image = left_image[40:-15,:]
                left_image = cv2.resize(left_image, (200, 66), interpolation=cv2.INTER_AREA)

                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)

                name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image = right_image[40:-15,:]
                right_image = cv2.resize(right_image, (200, 66), interpolation=cv2.INTER_AREA)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)

            augmented_images = []
            augmented_measurements = []

            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=120)
validation_generator = generator(validation_samples, batch_size=120)

ch, row, col = 3, 160, 320  # Trimmed image format

# As a first step LeNet model was used for the project and then moved on
# to Nvidia model

'''
#LeNet Model

model = Sequential()
model.add(Lambda(lambda x: x/ 255.0, input_shape=(66, 200, 3)))


model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

'''
Nvidia model
'''

model = Sequential()
#model.add(Cropping2D(cropping=((69, 25), (0,0)), input_shape=(160, 320, 3)))

#model.add(Lambda((lambda x: x/127.5 - 1.)))
model.add(Lambda(normalize,input_shape=(66,200,3)))
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))
#model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Conv2D(24, (5, 5), activation='relu', strides = (2, 2)))
#model.add(Dropout(0.1))
model.add(Conv2D(36, (5, 5), activation='relu', strides = (2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(48, (5, 5), activation='relu', strides = (2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
#model.add(Dropout(0.3))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))

# At times multi gpu model was used to speed up the training

#multi_model = multi_gpu_model(model, gpus=2)
print(model.summary())
for layer in model.layers:
    print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
model.compile(loss='mse', optimizer = 'adam')
#multi_model.compile(loss='mse', optimizer = 'adam')

#multi_model.fit(x = X_train, y = y_train, batch_size=2048, validation_split=0.2, shuffle=True, epochs = 5, verbose = 1)
'''model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
'''

b_size = 720
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//b_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)//b_size,
                    epochs=10, verbose = 1)

#model.set_weights(multi_model.get_weights())

# Saving the model

model.save('model_nvidia_bs720_4d30_7d40_10d50_e10_1.h5')

# Save model weights only
model.save_weights('./model_nvidia_weights_bs720_4d30_7d40_10d50_e10_1.h5')
json_string = model.to_json()
with open('./model_nvidia_bs720_4d30_7d40_10d50_e10_1.json', 'w') as f:
    f.write(json_string)
