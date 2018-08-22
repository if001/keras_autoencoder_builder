# keras autoencoder builder
簡単にautoencoderが作れます。
入力レイヤーと出力レイヤー1つの前提です。
複数入力への拡張は今後のお話。

```python
auto_encoder = AutoEncoderBuidler(
                 loss='mean_squared_error',
                 opt='adam',
                 encoder_weight_file_path=''
                 decoder_weight_file_path=''
	)
```

encoderとdecoderのweightを保存したファイルを渡すとモデルがロードされます。
