from parser import ReutersParser
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import utils

#target_topics = ['earn', 'acq', 'money-fx', 'crude', 'grain', 'trade', 'interest', 'ec', 'wheat', 'ship']
target_topics = ['earn', 'acq', 'money-fx', 'crude', 'grain']
target_places = ['usa', 'australia']
dict_topics = \
    {
    'earn': 1,
    'acq': 2,
    'money-fx': 3,
    'crude': 4,
    'grain': 5
    # 'trade': 6,
    # 'interest': 7,
    # 'ec': 8,
    # 'wheat': 9,
    # 'ship': 10
    }


types = ['training-set', 'published-testset']


def train(X, y, classifier_name):
    if classifier_name == "svm":
        classified_data = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    elif classifier_name == "perceptron":
        classified_data = Perceptron(random_state=123)
    classified_data.fit(X, y)

    return classified_data


def train_perceptron(ref_docs):
    x_train, y_train, vectorizer = utils.create_vectorized_data(ref_docs, "training-set", dict_topics,
                                                                'CountVectorizer')
    classified_data = Perceptron(random_state=123)
    classified_data.fit(x_train, y_train)

    corpus_test, y_test = utils.create_vectorized_data(ref_docs, "published-testset", dict_topics, 'CountVectorizer')
    x_test = vectorizer.transform(corpus_test)

    scores = classified_data.score(x_test, y_test)
    print(scores)

    pred = classified_data.predict(x_test)

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=target_topics))
    print(precision_recall_fscore_support(y_test, pred))


def train_bert(ref_docs):
    from bert_classifier import prepare_tf_data, build_model
    x_train, y_train, x_test, y_test, test_df = prepare_tf_data(ref_docs, dict_topics)
    model = build_model()
    train_history = model.fit(
        x={'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']},
        y=y_train,
        validation_data=(
            {'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']}, y_test
        ),
        epochs=1,
        batch_size=36
    )
    import numpy as np

    predicted_raw = model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})
    y_predicted = np.argmax(predicted_raw, axis=1)
    y_true = test_df.topic_d

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_predicted))


def train_multilabel(ref_docs):
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    import utils_multilabel
    from utils_multilabel import filter_docs, get_most_important_topics, get_most_important_places,\
        obtain_topic_tags, obtain_place_tags

    topics = obtain_topic_tags()
    places = obtain_place_tags()
    target_topics = get_most_important_topics(ref_docs, places)
    target_places = get_most_important_places(ref_docs, topics)

    docs = filter_docs(ref_docs, target_topics, target_places, types)

    x_train, y_train, vectorizer = utils_multilabel.create_vectorized_data(docs, "training-set", target_topics,
                                                               target_places, 'CountVectorizer')
    corpus_test, y_test = utils_multilabel.create_vectorized_data(docs, "published-testset", target_topics,
                                                                target_places, 'CountVectorizer')
    x_test = vectorizer.transform(corpus_test)


    clf = MultiOutputClassifier(LogisticRegression(max_iter = 1000)).fit(x_train, y_train)

    pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score
    print('Accuracy Score: ', accuracy_score(y_test, pred))

    from sklearn.metrics import hamming_loss
    print('Hamming Loss: ', round(hamming_loss(y_test, pred), 2))


def main(model):

    # Create the list of Reuters data and create the parser
    files = ["data/reut2-%03d.sgm" % r for r in range(0, 22)]
    parser = ReutersParser()

    print("Parsing training data...\n")
    docs = []
    for fn in files:
        for d in parser.parse(open(fn, 'rb')):
            docs.append(d)

    print(docs[0:2])

    places = utils.obtain_place_tags()
    topics = utils.get_most_important_topics(docs, places)
    ref_docs = utils.filter_doc_list_through_topics_train_test(topics, types, docs)

    if model=='perceptron':
        train_perceptron(ref_docs)
    elif model == 'bert':
        train_bert(ref_docs)
    elif model == 'multi_label':

        train_multilabel(docs)


if __name__ == "__main__":
    main(model='multi_label')

