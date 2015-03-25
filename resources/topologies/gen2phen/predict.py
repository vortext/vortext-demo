import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import sklearn
import json

import itertools
from itertools import chain, repeat, islice

import ner

from nltk.corpus import stopwords

import nltk.tokenize

from collections import defaultdict

from ftfy import fix_text


sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/gen2phen/"))

from mintz_model2 import tokenize, MintzModel

class Handler():

    title = "Genome-disease relations"

    model = None

    '''''
    These are lifted from `tmVar: A text mining approach for
    extracting sequence variants in biomedical literature` (Chih-Hsuan Wei et.al.)
    And match HGVS entities on a per-token basis. See Table 3 of their publication.
    '''''
    hgvs_regex = {
        "genomic_1": re.compile("([cgrm]\.[ATCGatcgu \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(inv|del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu0-9\_\.\:]*)"),
        "genomic_2": re.compile("(IVS[ATCGatcgu \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu0-9\_\.\:]*)"),
        "genomic_3": re.compile("([cgrm]\.[ATCGatcgu \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)"),
        "genomic_4": re.compile("(IVS[ATCGatcgu \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)"),
        "genomic_5": re.compile("([cgrm]\.[ATCGatcgu][0-9]+[ATCGatcgu])"),
        "genomic_6": re.compile("([ATCGatcgu][0-9]+[ATCGatcgu])"),
        "genomic_7": re.compile("([0-9]+(del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu]*)"),
        "protein_1": re.compile("([p]\.[CISQMNPKDTFAGHLRWVEYX \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(inv|del|ins|dup|tri|qua|con|delins|indel|fsX|fsx|fsx|fs)[CISQMNPKDTFAGHLRWVEYX \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]*)"),
        "protein_2": re.compile("([p]\.[CISQMNPKDTFAGHLRWVEYX \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)"),
        "protein_3": re.compile("([p]\.[A-Z][a-z]{0,2}[\W\-]{0,1}[0-9]+[\W\-]{0,1}[A-Z][a-z]{0,2})"),
        "protein_4": re.compile("([p]\.[A-Z][a-z]{0,2}[\W\-]{0,1}[0-9]+[\W\-]{0,1}(fs|fsx|fsX))")
    }

    def load_model(self, filename):
        logging.info("loading model %s" % (filename))
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def __init__(self):
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        self.hgvs_regexes = self.hgvs_regex.values()

        self.sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')

        logging.info("constructing model")
        self.model = MintzModel(script_dir);
        self.cls, self.vectorizer = self.load_model(os.path.join(script_dir, "models/model.pck"))

    def get_sentences(self, document):
        return self.sent_tokenize.span_tokenize(document, realign_boundaries=True)

    def handle(self, payload):
        document = json.loads(payload)

        document_text = " ".join(document["pages"])

        sents = self.get_sentences(document_text)

        annotations = []
        def annotation(sent, sent_text):
            return {"uuid": str(uuid.uuid1()),
                    "position": sent[0],
                    "prefix": document_text[max(sent[0] - 32, 0):sent[0]],
                    "suffix": document_text[sent[1]:min(sent[1] + 32, len(document_text))],
                    "content": sent_text}

        predictions = defaultdict(set)

        for sent in sents:
            sent_text = document_text[sent[0]:sent[1]]

            s = fix_text(unicode(sent_text))

            included, excluded = tokenize(s)
            words = [t.norm_ for t in included]

            entities = self.model.get_entities(words, use_ner=True)

            if "DISEASE" in entities and "GENE" in entities and len(included):
                feature_dict = self.model.get_feature_dicts(s, included, excluded, entities)
                if not feature_dict:
                    continue
                X = self.vectorizer.transform(feature_dict)
                predict = self.cls.predict(X)

                if any(predict):
                    pairs = itertools.product(entities["GENE"], entities["DISEASE"])
                    for pair in pairs:
                        gene = " ".join(words[pair[0][0]:pair[0][1]])
                        disease = " ".join(words[pair[1][0]:pair[1][1]])
                        k = gene + " - " + disease
                        predictions[k] = predictions[k].union([(sent, sent_text)])


        output = []
        for key, annotations in reversed(sorted(predictions.items(), key=lambda x: len(x[1]))):
            out = {
                "type": "Genome-disease relation prediction",
                "title": key,
                "description": "Possible association",
                "annotations": [annotation(sent,sent_text) for sent, sent_text in annotations]
            }
            output.append(out)

        return json.dumps({"marginalia": output})
