from keras.models import Model
from layer_set import LayerSet


class AutoEncoderBuidler():
    def __init__(self,
                 loss='mean_squared_error',
                 opt='adam',
                 encoder_weight_file=''
                 decoder_weight_file=''
                 ):
        """
        loss: オブジェクト or String
        optimizer: オブジェクト or String
        encoder_weight_file_path: ファイル名が指定されていればモデルをロード
        decoder_weight_file_path: ファイル名が指定されていればモデルをロード
        """

        self.__loss = AutoEncoderBuidler.open_func(loss)
        self.__opt = AutoEncoderBuidler.open_func(opt)

        self.encoder = self.__model_builder(
            LayerSet.encoder_input(),
            [LayerSet.encoder_layer],
            encoder_weight_file_path
        )

        self.decoder = self.__model_builder(
            LayerSet.decoder_input(),
            [LayerSet.decoder_layer],
            decoder_weight_file_path
        )

        self.auto_encoder = self.__build_from_model(
            LayerSet.encoder_input(),
            [self.encoder, self.decoder]
        )

    @classmethod
    def open_func(cls, f):
        """
        optimizerやlossが、オブジェクト or Stringで飛んでくるため
        ここで開けて関数にする
        """
        return f if type(f) == str else f()

    def __model_builder(self, input_layer, layers, file_path=''):
        if file_path == '':
            __first_inp = input_layer
            for layer in layers:
                output_layer = layer(input_layer)
                input_layer = output_layer
            model = Model(__first_inp, output_layer)
            model.compile(optimizer=self.__opt, loss=self.__loss)
            model.summary()
        else:
            model = self.load(weight_file_path)
        return model

    def __build_from_model(self, input_layer, models):
        __concat_layers = []
        for model in models:
            __concat_layers += model.layers
        return self.__model_builder(input_layer, __concat_layers)


def main():
    builded = AutoEncoderBuidler()

    builded.encoder.fit()
    builded.decoder.fit()
    builded.auto_encoder.fit()
    builded.auto_encoder.predict()


if __name__ == "__main__":
    main()
