def load_tokenizer_and_model():
    from tensorflow import keras
    import pickle as pkl

    with open('./models/char_level_chinese_tokenizer_5000.pkl', 'rb') as f:
      tokenizer = pkl.load(f)

    model = keras.models.load_model('./models/plain_lstm.hdf5')

    return tokenizer, model


tokenizer, model = None, None


def make_prediction(sent: str) -> float:
  """ Returns a score between 0 and 1, with 1 being positive """
  from tensorflow import keras

  global tokenizer, model

  if tokenizer is None:
    tokenizer, model = load_tokenizer_and_model()

  encoded = tokenizer.texts_to_sequences([sent])[0]
  encoded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=50, padding='pre')
  return model.predict(encoded)[0][1]
