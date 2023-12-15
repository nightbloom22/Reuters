import re
from html.parser import HTMLParser

class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.
    """
    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.in_reuters = False
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.reuters = ""
        self.cgisplit = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))

            for doc in self.docs:
                yield doc
                #print('-----', doc)
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag".
        """
        if tag == "reuters":

            self.in_reuters = True
            for attribute in attrs:
                if attribute[0] == "cgisplit":
                    tes = str((attribute[1].encode("utf-8")).lower())[1:]
                    tes = tes.replace("'", "")
                    self.cgisplit = tes
                    break
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag".
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.in_reuters = False
            self.docs.append( (self.topics, self.body, self.cgisplit) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            #print(str(data.encode("utf-8")))
            tes = str(data.encode("utf-8"))
            tes = tes.replace("b'", '')
            tes = tes.replace("b''", '')
            tes = tes.replace("\\n'", ' ')
            tes = tes.replace("\\n", ' ')
            tes = tes.replace("'", '')

            self.body += tes #str(data.encode("utf-8"))
        elif self.in_topic_d:
            #print(str(data.encode("utf-8")))
            tes = str(data.encode("utf-8"))
            tes = tes.replace("b'", '')
            tes = tes.replace("b''", '')
            tes = tes.replace("\\n'", ' ')
            tes = tes.replace("\\n", ' ')
            tes = tes.replace("'", '')

            self.topic_d += tes  # str(data.encode("utf-8"))
            #self.topic_d += str(data.encode("utf-8"))

