import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import sklearn
import json

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/ebm/"))
import quality3

class Handler():

    title = "PICO"

    PICO_DOMAINS = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]
    PICO_TITLES = {"CHAR_PARTICIPANTS": "Population",
                   "CHAR_INTERVENTIONS": "Intervention",
                   "CHAR_OUTCOMES": "Outcomes"}

    models = {}

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def vectorize(self, sentences):
        h = HashingVectorizer(stop_words=stopwords.words('english'),
                              norm="l2",
                              ngram_range=(1, 2),
                              analyzer="word",
                              strip_accents="ascii",
                              decode_error="ignore")
        return h.transform(sentences)

    def __init__(self):
        for domain in self.PICO_DOMAINS:
            script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
            rel_path = "models/" + domain + ".pck"
            models_file = os.path.join(script_dir, rel_path)
            log.info("%s: loading models: %s" % (self.title, models_file))
            self.models[domain] = self.load_model(models_file)
            log.info("%s: done loading models" % (self.title))

        self.sentence_tokenizer = PunktSentenceTokenizer()

    def handle(self, payload):
        document = json.loads(payload)

        # first get sentence indices in full text
        document_text = " ".join(document["pages"])

        sent_text = self.sentence_tokenizer.tokenize(document_text)

        X_sents = self.vectorize(sent_text)

        output = []
        for domain in self.PICO_DOMAINS:
            pred_sents = [int(x_i) for x_i in self.models[domain].predict(X_sents)]

            positive_sents = [sent for (sent, pred) in zip(sent_text, pred_sents) if pred == 1]

            annotations = []
            for sent in positive_sents:
                annotations += [{"content": sent, "uuid": str(uuid.uuid1()), "label": "biased"}]

            output.append({
                "annotations": annotations,
                "description": "**This is *very* experimental**",
                "title": self.PICO_TITLES[domain],
                "type": self.title
            })

        return json.dumps({"marginalia": output})
