
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


def obtain_place_tags():
    """
    Open the topic list file and import all of the topic names taking care to strip the trailing "\n" from each word.
    """
    places = open("data/all-places-strings.lc.txt", "r").readlines()
    places = [t.strip() for t in places]
    for i in range(0, len(places)):
        places[i] = places[i].lower()
    return places


def filter_docs(docs, topics, places, types):
    ref_docs_1 = []
    for d in docs:
        labels = d[0]

        top_l = []
        for label in labels:

            if label in topics:
                top_l.append(label)
        d_tup = (labels, d[1], d[2], top_l)
        ref_docs_1.append(d_tup)

    ref_docs = []
    for dp in ref_docs_1:
        labels = dp[0]
        l = dp[3]
        for label in labels:
            if label in places:
                l.append(label)
        d_tup = (l, dp[1], dp[2])
        ref_docs.append(d_tup)

    return ref_docs


def get_most_important_topics(docs, places):
    topic_list = []
    for doc in docs:
        topic_list.extend(doc[0])
    topic_list = [i for i in topic_list if i not in places]

    from collections import Counter

    data = Counter(topic_list)

    topics = []
    for i in data.most_common(5):
        topics.append(i[0])

    return topics


def get_most_important_places(docs, topics):
    place_list = []
    for doc in docs:
        place_list.extend(doc[0])
    place_list = [i for i in place_list if i not in topics]

    from collections import Counter

    data = Counter(place_list)

    places = []
    for i in data.most_common(5):
        places.append(i[0])

    return places


def create_vectorized_data(docs, doc_type, topics, places, vectorizer):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list.

    The function returns both the class label vector (y) and
    the corpus token/feature matrix (X).
    """

    docs = [doc for doc in docs if doc[2] == doc_type]
    docs =[doc for doc in docs if doc[0] != []]
    docs = [doc for doc in docs if doc[1] != []]

    y_ = [d[0] for d in docs if d[0] != []]
    print(y_)

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_)

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    if doc_type != "training-set":
        return corpus, y

    if vectorizer == 'CountVectorizer':
        vectorizer = vectorization.StemmedCountVectorizer(stop_words="english", analyzer="word", min_df=1, max_features=1000)
    elif vectorizer == 'TfidfVectorizer':
        vectorizer = vectorization.StemmedTfidfVectorizer(stop_words="english", analyzer="word", min_df=1, max_features=1000)

    if doc_type == "training-set":
        x = vectorizer.fit_transform(corpus)
        return x, y, vectorizer


