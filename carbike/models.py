from django.db import models

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64


graph = tfv1.get_default_graph()
class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224 #画像サイズ
    MODEL_FILE_PATH = './carbike/ml_models/VGG16_transfer.h5' #モデルファイルパス
    classes = ["car", "motorbike"]
    num_classes = len(classes)
    
    def predict(self):
        model = None
        global graph

        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            #get uploaded image
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)
            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE))
            data = np.asarray(image)/255.0
            X = []
            X.append(data)
            X = np.array(X)

            #predict
            result = model.predict([X])[0]
            predicted = result.argmax()  #最大値のindex
            percentage = int(result[predicted] * 100)
            #print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage
    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' +  img.file.content_type + ';base64,' + base64_img



