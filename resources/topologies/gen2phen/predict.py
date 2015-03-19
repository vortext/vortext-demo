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


nlp = spacy.en.English()

from spacy.parts_of_speech import DET, NUM, PUNCT, X, PRT, NO_TAG, EOL
def is_eligable(tok):
    exclude = [DET, NUM, PUNCT, X, PRT, NO_TAG, EOL]
    return tok.pos not in exclude and tok.norm_.isalnum()

def tokenize(s):
    included = []
    excluded = []
    for tok in nlp(s, parse=False, tag=True):
        if is_eligable(tok):
            included.append(tok)
        else:
            excluded.append(tok)
    return included, excluded


sparse = re.compile("\s{3,}")
def is_sparse(sent):
    return True if re.search(sparse, sent) else False


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
        self.model, self.vectorizer = self.load_model(os.path.join(script_dir, "models/model.pck"))

    def get_sentences(self, document):
        return self.sent_tokenizer.span_tokenize(document, realign_boundaries=True)

    def get_entities(self, sent):
        entities = defaultdict(list)
        entities.update(self.ner_tagger.get_entities(sent))
        for tok in sent.split():
            if any([re.match(expr, tok) for expr in self.hgvs_regexes]):
                entities["GENE"].append(tok)
        return entities

    def pad_infinite(self, iterable, padding=None):
        return chain(iterable, repeat(padding))

    def pad(self, iterable, size, padding=None):
        return islice(self.pad_infinite(iterable, padding), size)

    def get_feature_dicts(self, sent, entities):
        tokens, excluded = tokenize(sent)
        is_junk = (float(len(excluded)) / float(len(tokens))) > 0.8

        token_strings = [tok.norm_ for tok in tokens]
        pos = [tok.norm_ + "/" + tok.pos_ for tok in tokens]

        gene_tokens = [([tok.norm_ for tok in tokens], "GENE") for tokens in [nlp(concept, parse=False, tag=False) for concept in entities["GENE"]]]
        disease_tokens = [([tok.norm_ for tok in tokens], "DISEASE") for tokens in [nlp(concept, parse=False, tag=False) for concept in entities["DISEASE"]]]

        pairs = [sorted(pair, key=lambda x: x[0]) for pair in itertools.product(gene_tokens, disease_tokens)]


        features = []
        for pair in pairs:
            for window in range(0,3):
                feature_dict = {
                    "sparse": is_sparse(sent),
                    "junk": is_junk,
                    "first": pair[0][1],
                    "second": pair[1][1]}

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

                except Exception:
                    continue
        return features



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
            entities = {}

            s = fix_text(unicode(sent_text))
            try:
                entities = self.get_entities(s)
            except Exception, e:
                logging.error(e)
                continue

            if "DISEASE" in entities and "GENE" in entities:
                feature_dict = self.get_feature_dicts(s, entities)
                if not feature_dict:
                    continue
                X = self.vectorizer.transform(feature_dict)
                predict = self.model.predict(X)
                if any(predict):
                    pairs = itertools.product(entities["GENE"], entities["DISEASE"])

                    for pair in pairs:
                        k = pair[0] + " - " + pair[1]
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
