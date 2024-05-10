import tensorflow as tf

'''Our sequential model is defined below which process the output of resnet to get generate accurate
predictions'''
class TumorClassifierModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TumorClassifierModel,self).__init__(**kwargs)
        self.added_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(420),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(220),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(40),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])
    def call(self,inputs):
        return self.added_layers(inputs)