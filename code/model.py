import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization


class TumorClassifierModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TumorClassifierModel,self).__init__(**kwargs)
        self.added_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16,(3,3),strides=(2,2),padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(420),
            # tf.keras.layers.LeakyReLU(),
            # tf.keras.layers.Dense(80),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])
    def call(self,inputs):
        return self.added_layers(inputs)
    
# def build_model(input_shape=(224, 224, 3)):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

#     for layer in base_model.layers:
#         layer.trainable = False

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)  
#     x = BatchNormalization()(x)      
#     x = Dropout(0.5)(x)              
#     x = Dense(1024, activation='relu')(x)  
#     x = BatchNormalization()(x)     
#     x = Dropout(0.5)(x)              

#     predictions = Dense(1, activation='sigmoid')(x)

#     model = Model(inputs=base_model.input, outputs=predictions)

#     return model

# if __name__ == '__main__':
#     model = build_model()
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()