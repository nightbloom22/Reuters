
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")


def prepare_tf_data(ref_docs, dict_topics):
    raw_df = pd.DataFrame(ref_docs, columns=['topic', 'text', 'tags'])

    train_df = raw_df[raw_df['tags']=='training-set']
    train_df = train_df[['text', 'topic']]
    train_df['topic_d'] = train_df.topic.map(dict_topics)

    test_df = raw_df[raw_df['tags']=='published-testset']
    test_df = test_df[['text', 'topic']]
    test_df['topic_d'] = test_df.topic.map(dict_topics)
    print('------', test_df.topic.unique())

    y_train = to_categorical(train_df.topic_d)
    y_test = to_categorical(test_df.topic_d)
    print(y_test)

    x_train = tokenizer(
        text=train_df.text.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    x_test = tokenizer(
        text=test_df.text.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    return x_train, y_train, x_test, y_test, test_df


def build_model():
    max_len = 70
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    embeddings = bert(input_ids,attention_mask = input_mask)[0]
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(32,activation = 'relu')(out)
    y = Dense(11,activation = 'sigmoid')(out)
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True

    optimizer = Adam(
        learning_rate=5e-05,
        epsilon=1e-08,
        # decay=0.01,
        clipnorm=1.0)
    # Set loss and metrics
    loss =CategoricalCrossentropy(from_logits = True)
    metric = CategoricalAccuracy('balanced_accuracy'),
    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metric)
    return model

