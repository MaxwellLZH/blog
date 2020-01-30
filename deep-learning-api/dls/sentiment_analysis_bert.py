import numpy as np


def load_tokenizer_and_model():
    # move heavy imports into the load function
    from transformers import BertConfig, TFBertForSequenceClassification, BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    bert_config = BertConfig.from_dict({
                                      "attention_probs_dropout_prob": 0.1, 
                                      "directionality": "bidi", 
                                      "hidden_act": "gelu", 
                                      "hidden_dropout_prob": 0.1, 
                                      "hidden_size": 768, 
                                      "initializer_range": 0.02, 
                                      "intermediate_size": 3072, 
                                      "max_position_embeddings": 512, 
                                      "num_attention_heads": 12, 
                                      "num_hidden_layers": 12, 
                                      "pooler_fc_size": 768, 
                                      "pooler_num_attention_heads": 12, 
                                      "pooler_num_fc_layers": 3, 
                                      "pooler_size_per_head": 128, 
                                      "pooler_type": "first_token_transform", 
                                      "type_vocab_size": 2, 
                                      "vocab_size": 21128
                                    })
    model = TFBertForSequenceClassification(bert_config)
    # Call the model once so it gets build with the right shape before loading the weights
    model((np.random.random((1, 100)) * 50).astype(int))
    model.load_weights('./models/sentiment_analysis_bert.hdf5')
    return tokenizer, model


tokenizer, model = None, None


def make_prediction(sent: str) -> float:
    from tensorflow import keras

    global tokenizer, model
    if tokenizer is None:
        # lazy loading
        tokenizer, model = load_tokenizer_and_model()

    encoded = tokenizer.encode(sent, max_length=20, add_special_tokens=True)
    inference = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=20, padding='post')
    inference = np.array(inference)

    pred = model.predict(inference)[0]
    # convert to a score between 0 and 1, with 1 being positive and 0 negative
    pred = np.exp(pred)[1] / np.exp(pred).sum()
    return pred
