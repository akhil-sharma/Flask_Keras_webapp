from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from model.load import init

app = Flask(__name__)

UPLOAD_FOLDER = 'UPLOADS'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# classifier = init()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/')
def index():
    return render_template("cd_home.html")


@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return "NoFileError"
        image_data = request.files['file']
        if image_data.filename == '':
            return "IncorrectTypeError"
        if image_data and allowed_file(image_data.filename):
            filename = secure_filename(image_data.filename)
            filename = "upload" + "." + filename.rsplit('.', 1)[1]
            save_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_data.save(save_url)
            test_image = image.load_img(save_url, target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            result = classifier.predict(test_image)
            if result[0][0] == 1:
                prediction = 'dog'
            else:
                prediction = 'cat'

            return prediction


if __name__ == '__main__':
    app.run()
