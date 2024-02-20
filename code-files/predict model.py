import cv2
import numpy as np
from keras.models import load_model

from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import h5py
# Load the saved model
model = load_model('model_hand.h5')

# Load the image you want to predict
img = cv2.imread('C:\\Users\\VK\\Desktop\\miniproject\\d.jpg', cv2.IMREAD_GRAYSCALE)


# Resize the image to 28x28 and invert the colors
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)

# Reshape the image to a 4D tensor (batch_size, height, width, channels)
img = img.reshape((1, 28, 28, 1))

# Make the prediction
prediction = model.predict(img)

# Get the predicted label
predicted_label = np.argmax(prediction)

# Print the predicted label
print("Predicted label:", chr(65 + predicted_label))