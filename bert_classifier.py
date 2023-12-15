
# from parser import ReutersParser
# import utils
# dict_topics = \
#     {
#     'earn': 1,
#     'acq': 2,
#     'money-fx': 3,
#     'crude': 4,
#     'grain': 5,
#     'trade': 6,
#     'interest': 7,
#     'ec': 8,
#     'wheat': 9,
#     'ship': 10
#     }
#
# types = ['training-set', 'published-testset']
# files = ["data/reut2-%03d.sgm" % r for r in range(0, 22)]
# parser = ReutersParser()
#
# print("Parsing training data...\n")
# docs = []
# for fn in files:
#     for d in parser.parse(open(fn, 'rb')):
#         docs.append(d)
#
# places = utils.obtain_place_tags()
#
# topics = utils.get_most_important_topics(docs, places)
# print('--topics--', topics)
#
# ref_docs = utils.filter_doc_list_through_topics_train_test(topics, types, docs)


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
        learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website
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


# x_train, y_train, x_test, y_test, test_df = prepare_tf_data(ref_docs)
# model = build_model()
# train_history = model.fit(
#     x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
#     y = y_train,
#     validation_data = (
#     {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
#     ),
#   epochs=1,
#     batch_size=36
# )
#
# import numpy as np
#
# predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
# y_predicted = np.argmax(predicted_raw, axis = 1)
# y_true = test_df.topic_d
#
# from sklearn.metrics import classification_report
# print(classification_report(y_true, y_predicted))



