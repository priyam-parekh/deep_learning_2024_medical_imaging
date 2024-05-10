import tensorflow as tf

'''Our sequential model is defined below which process the image_array to get generate accurate
predictions'''

class TumorClassifierModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TumorClassifierModel,self).__init__(**kwargs)
        self.added_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,kernel_size=(7,7),strides=(1,1),padding='same',input_shape=(224,224,1)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32,kernel_size=(5,5),strides=(1,1),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(520),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(240),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(140),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(1,activation='sigmoid')

        ])
    def call(self,inputs):
        return self.added_layers(inputs)