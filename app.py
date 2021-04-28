from tensorflow import keras
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('model_resnet_50.h5')

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'img_uploads', secure_filename(file.filename))
        file.save(file_path)

        image = load_img(file_path, target_size=(224, 224))
        image_arr = img_to_array(image)
        image_arr = image_arr/255
        x = np.expand_dims(image_arr, axis=0)
        pred = model.predict(x)
        pred = np.argmax(pred, axis=1)
        pred = get_prediction(pred)
    return pred

def get_prediction(pred):
    if pred==0:
        pred="The leaf is diseased tomato leaf"
    elif pred==1:
        pred="It is diseased tomato plant"
    elif pred==2:
        pred="The leaf is fresh tomato leaf"
    else:
        pred="It is fresh tomato plant"
    return pred

