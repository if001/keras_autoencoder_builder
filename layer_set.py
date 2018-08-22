from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import concatenate

from keras.layers.wrappers import TimeDistributed as TD


class LayerSet():
    @classmethod
    def encoder_input(cls):
        return Input(shape=(28, 28, 3))

    @classmethod
    def multiple_encoder_input(cls):
        input1 = Input(shape=(28, 28, 3))
        input2 = Input(shape=(28, 28, 3))
        output_layer = concatenate([input1, input2])
        return output_layer

    @classmethod
    def encoder_layer(cls, input_layer):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        return encoded

    @classmethod
    def decoder_input(cls):
        return Input(shape=(4, 4, 8))

    @classmethod
    def decoder_layer(cls, input_layer):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        return decoded
