import pandas as pd
import vectorization


def obtain_topic_tags():
    """
    Open the topic list file and import all of the topic names taking care to strip the trailing "\n" from each word.
    """
    topics = open("data/all-topics-strings.lc.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    for i in range(0, len(topics)):
        topics[i] = topics[i].lower()

    return topics


def filter_doc_list_through_topics_train_test(topics, types, docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single feature entry and the body text, instead of
    a list of topics. It removes all geographic features and only
    retains those documents which have at least one non-geographic topic.
    """
    ref_docs = []

    for d in docs:

        if d[0] == [] or d[0] == "":
            continue
        if str(d[2]) not in types:
            continue
        labels = d[0]

        for label in labels:
            if label in topics:
                d_tup = (label, d[1], d[2])
                ref_docs.append(d_tup)
                break
    return ref_docs


def obtain_place_tags():
    """
    Open the topic list file and import all of the topic names taking care to strip the trailing "\n" from each word.
    """
    topics = open("data/all-places-strings.lc.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    for i in range(0, len(topics)):
        topics[i] = topics[i].lower()
    return topics


def get_most_important_topics(docs, places):
    topic_list = []
    for doc in docs:

        topic_list.extend(doc[0])
    topic_list = [i for i in topic_list if i not in places]

    from collections import Counter

    data = Counter(topic_list)

    topics = []
    for i in data.most_common(10):
        topics.append(i[0])
    #topics = [i for i in topics if i != 'wheat']

    return topics


def create_vectorized_data(docs, doc_type, dict_topics, vectorizer):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list.

    The function returns both the class label vector (y) and
    the corpus token/feature matrix (X).
    """
    docs = [doc for doc in docs if doc[2] == doc_type]

    y_ = [d[0] for d in docs]
    y = pd.DataFrame(y_, columns=['y_'])
    y['y_'] = y['y_'].map(dict_topics)

    y = y['y_'].values.tolist()

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    if vectorizer == 'CountVectorizer':
        vectorizer = vectorization.StemmedCountVectorizer(stop_words="english", analyzer="word", min_df=1)
    elif vectorizer == 'TfidfVectorizer':
        vectorizer = vectorization.StemmedTfidfVectorizer(stop_words="english", analyzer="word", min_df=1)

    if doc_type == "training-set":
        x = vectorizer.fit_transform(corpus)
        return x, y, vectorizer
    else:
        return corpus, y