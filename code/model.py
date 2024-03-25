import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class ClassificationModel:
    model = None
    target = None

    def load_model(self):
        path = os.path.join(os.getcwd(), 'code/sportsEN_V2_0219_2.h5')
        model = tf.keras.models.load_model(path, custom_objects={'KerasLayer':hub.KerasLayer})
        path = os.path.join(os.getcwd(), 'code/class_label.json')
        with open(path) as f:
            class_label = json.load(f)
        self.model = model
        self.targets = class_label  
    
    def predict(self, image, n) -> list:
        im = tf.constant(image[:, :, :3])      
        im = tf.expand_dims(im, axis = 0)

        if tf.test.gpu_device_name() == '' or tf.test.gpu_device_name() == None:
            pred = self.model.predict(im, verbose = 0)
        else:
            with tf.device(tf.test.gpu_device_name()):
                pred = self.model.predict(im, verbose = 0)
        
        dtype = [('class_index', int), ('prob', float)]
        value = [(int(i), pred[0][i]) for i in range(len(pred[0]))]
        results = np.array(value, dtype=dtype)
        results = np.sort(results, order='prob')[::-1]
        return {self.targets[str(result[0])]: result[1] for result in results[:n]}

