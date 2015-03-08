import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import sklearn
import json

import ner

from nltk.corpus import stopwords

import nltk.tokenize

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
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def __init__(self):
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        self.ner_tagger = ner.SocketNER(host='localhost', port=9191)

    def handle(self, payload):
        document = json.loads(payload)

        document_text = " ".join(document["pages"])

        annotations = []

        sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenize.span_tokenize(document_text,realign_boundaries = True)

        word_tokenizer = nltk.tokenize.regexp.WhitespaceTokenizer()

        def add_annotation(sent, sent_text):
            annotations.append({"uuid": str(uuid.uuid1()),
                                "position": sent[0],
                                "prefix": document_text[max(sent[0] - 32, 0):sent[0]],
                                "suffix": document_text[sent[1]:min(sent[1] + 32, len(document_text))],
                                "content": sent_text})

        for sent in sents:
            sent_text = document_text[sent[0]:sent[1]]
            entities = {}

            hgvs_regexes = self.hgvs_regex.values()

            try:
                entities = self.ner_tagger.get_entities(sent_text.encode("utf-8", errors="ignore"))
            except Exception, e:
                logging.error(e)
                continue

            if "DISEASE" in entities and "GENE" in entities:
                logging.info(entities)
                add_annotation(sent, sent_text)
            elif "DISEASE" in entities:
                words = word_tokenizer.tokenize(sent_text)
                logging.debug("disease but no gene, checking for hgvs in %s" % sent_text)
                if any([any([re.match(expr, word) for word in words]) for expr in hgvs_regexes]):
                    logging.info("adding %s" % sent_text)
                    add_annotation(sent, sent_text)

        output = [{
            "annotations": annotations,
            "description": "**This is *very* experimental**",
            "title": "gen2phen",
            "type": self.title
        }]

        return json.dumps({"marginalia": output})
