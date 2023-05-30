import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten

# Define the YOLO model
def YOLO(input_shape, num_classes):
    # Input layer
    input_tensor = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=x)
    return model

# Set the input shape and number of classes
input_shape = (224, 224, 3)  # Example shape, adjust as per your requirements
num_classes = 10  # Example number of classes, adjust as per your dataset

# Create the YOLO model
model = YOLO(input_shape, num_classes)

# Print the model summary
model.summary()
