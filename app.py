from flask import Flask, render_template, request, jsonify
import base64
import os
from flask import Flask,request,render_template
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image



UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"
labels = ['uang dua ribu','uang lima puluh ribu',
 'uang lima ribu',
 'uang sepuluh ribu',
 'uang seratus ribu',
 'uang seribu',
 'uang dua puluh ribu']

model = load_model("modellagi.h5", compile=False)
model.compile()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.json
    photo_data_url = data['photoDataUrl'].split(',')[1]  
    photo_bytes = base64.b64decode(photo_data_url)
    save_dir = 'static/uploads'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'capture.jpg')
    with open(file_path, 'wb') as f:
        f.write(photo_bytes)
    return jsonify({'message': 'Image saved successfully'})

@app.route('/predict',methods=['POST'])
def remback():
        file = 'capture.jpg'
        image = keras.preprocessing.image.load_img(UPLOAD_FOLDER +"/"+file, target_size=(150, 150))
        x = keras.preprocessing.image.img_to_array(image)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=32)
        preds = labels[np.argmax(pred) - 1]
        akurasi = np.round(np.max(pred), 2)
        classify = preds
        audio_src= "static/audio"
            
        if classify == "uang dua ribu":
            audio_src = "static/audio/duaribu.mp3"
        elif classify == "uang lima puluh ribu":
            audio_src = "static/audio/limapuluh.mp3"  
        elif classify == "uang lima ribu":
            audio_src = "static/audio/limaribu.mp3"
        elif classify == "uang sepuluh ribu":
            audio_src = "static/audio/sepuluhribu.mp3"  
        elif classify == "uang seratus ribu":
            audio_src = "static/audio/seratus.mp3"
        elif classify == "uang seribu":
            audio_src = "static/audio/seribu.mp3" 
        elif classify == "uang dua puluh ribu":
            audio_src = "static/audio/duapuluh.mp3"
        else:
            audio_src= None
        return render_template('index.html',file = file, akurasi =akurasi, prediction=classify, audio_src=audio_src) 
if __name__ == '__main__':
    app.run(debug=True)
    
    



