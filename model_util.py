"""
Model utility for image feature extraction using MobileNet.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Fix SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np

# TensorFlow 2.16+ uses keras 3.x with updated imports
try:
    from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
    from tensorflow.keras.preprocessing import image as process_image
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras import Model
except ImportError:
    # Fallback for older TensorFlow versions
    from tensorflow.python.keras.applications.mobilenet import MobileNet, preprocess_input
    from tensorflow.python.keras.preprocessing import image as process_image
    from tensorflow.python.keras.layers import GlobalAveragePooling2D
    from tensorflow.python.keras import Model


class DeepModel:
    """MobileNet deep model for feature extraction."""
    
    def __init__(self):
        self._model = self._define_model()
        print('✓ MobileNet model loaded successfully')
    
    @staticmethod
    def _define_model(output_layer=-1):
        """Define a pre-trained MobileNet model.
        
        Args:
            output_layer: the number of layer that output.
            
        Returns:
            Class of keras model with weights.
        """
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    @staticmethod
    def preprocess_image(path):
        """Process an image to numpy array.
        
        Args:
            path: the path of the image or BytesIO object.
            
        Returns:
            Numpy array of the image.
        """
        img = process_image.load_img(path, target_size=(224, 224))
        x = process_image.img_to_array(img)
        x = preprocess_input(x)
        return x
    
    @staticmethod
    def cosine_distance(input1, input2):
        """Calculate cosine distance between two feature sets.
        
        The return values lie in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.
        
        Args:
            input1, input2: two input numpy arrays.
            
        Returns:
            Element-wise cosine distances of two inputs.
        """
        return np.dot(input1, input2.T) / \
               np.dot(np.linalg.norm(input1, axis=1, keepdims=True),
                      np.linalg.norm(input2.T, axis=0, keepdims=True))
    
    def extract_feature(self, images):
        """Extract deep features using MobileNet model.
        
        Args:
            images: numpy array of preprocessed images.
            
        Returns:
            The output features of all inputs.
        """
        features = self._model.predict(images, verbose=0)
        return features

