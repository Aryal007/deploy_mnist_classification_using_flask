# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imread, imresize
# for matrix math
import numpy as np
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os

# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))

from load import *

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()

import base64


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = imread('output.png', mode='L')
    # make it the right size
    x = imresize(x, (28, 28))
    # imsave('final_image.jpg', x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])


if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=9091)
# optional if we want to run in debugging mode
# app.run(debug=True)
