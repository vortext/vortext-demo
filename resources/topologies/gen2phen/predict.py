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

import spacy.en

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/gen2phen/"))

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
        self.ner_tagger = ner.SocketNER(host='localhost', port=9191)
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.hgvs_regexes = self.hgvs_regex.values()
        self.nlp = spacy.en.English()
        self.vectorizer = self.load_model(os.path.join(script_dir, "models/vectorizer.pck"))
        self.model = self.load_model(os.path.join(script_dir, "models/model.pck"))

    def get_sentences(self, document):
        return self.sent_tokenizer.span_tokenize(document, realign_boundaries=True)

    def get_entities(self, sent):
        entities = defaultdict(list)
        entities.update(self.ner_tagger.get_entities(sent))
        for tok in sent.split():
            if any([re.match(expr, tok) for expr in self.hgvs_regexes]):
                entities["GENE"].append(tok)
        return entities

    def get_tokens(self, sent):
        return self.nlp(sent, tag=True, parse=False)

    def pad_infinite(self, iterable, padding=None):
        return chain(iterable, repeat(padding))

    def pad(self, iterable, size, padding=None):
        return islice(self.pad_infinite(iterable, padding), size)

    def entity_tokens(self, entities, t):
        return [([tok.norm_ for tok in tokens], t) for tokens in [self.nlp(concept, parse=False, tag=True) for concept in entities[t]]]

    def get_feature_dicts(self, tokens, entities):
        token_strings = [tok.norm_ for tok in tokens]
        pos = [tok.norm_ + "/" + tok.pos_ for tok in tokens]

        gene_tokens = self.entity_tokens(entities, "GENE")
        disease_tokens = self.entity_tokens(entities, "DISEASE")

        pairs = [sorted(pair, key=lambda x: x[0]) for pair in itertools.product(gene_tokens, disease_tokens)]

        features = []
        for pair in pairs:
            for window in range(0,3):
                feature_dict = {"first": pair[0][1], "second": pair[1][1]}

                try:
                    start = min(token_strings.index(pair[0][0][-1]) + 1, len(token_strings))
                    stop = token_strings.index(pair[1][0][0])
                    bound = sorted((start, stop))

                    feature_dict["pos"] = " ".join(pos[bound[0]:bound[1]])

                    left_window = token_strings[max(0, start - window - 1):start]
                    right_window = token_strings[stop + 1:min(len(token_strings) - 1, stop + window + 1)]

                    feature_dict["left"] = " ".join(self.pad(left_window, window, "#PAD#"))
                    feature_dict["right"] = " ".join(self.pad(right_window, window, "#PAD"))

                    # FIXME: add basic syntactic features (lift more from Mintz paper)
                    features.append(feature_dict)

                except Exception, e:
                    continue
        return features


    def handle(self, payload):
        document = json.loads(payload)

        document_text = " ".join(document["pages"])

        annotations = []

        sents = self.get_sentences(document_text)

        def add_annotation(sent, sent_text):
            annotations.append({"uuid": str(uuid.uuid1()),
                                "position": sent[0],
                                "prefix": document_text[max(sent[0] - 32, 0):sent[0]],
                                "suffix": document_text[sent[1]:min(sent[1] + 32, len(document_text))],
                                "content": sent_text})

        for sent in sents:
            sent_text = document_text[sent[0]:sent[1]]
            references = ["References", "references", "REFERENCES", "R E F E R E N C E S"]
            if any(s in sent_text for s in references):
                # Don't bother below this point, this assumes the pdf is linearized, but whatever
                break
            entities = {}

            try:
                entities = self.get_entities(sent_text)
            except Exception, e:
                logging.error(e)
                continue

            if "DISEASE" in entities and "GENE" in entities:
                tokens = self.get_tokens(sent_text)
                feature_dict = self.get_feature_dicts(tokens, entities)
                if not feature_dict:
                    continue
                X = self.vectorizer.transform(feature_dict)
                predict = self.model.predict(X)
                if any(predict):
                    add_annotation(sent, sent_text)

        output = [{
            "annotations": annotations,
            "description": "**This is *very* experimental**",
            "title": "gen2phen",
            "type": self.title
        }]

        return json.dumps({"marginalia": output})
