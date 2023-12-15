import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(CountVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))