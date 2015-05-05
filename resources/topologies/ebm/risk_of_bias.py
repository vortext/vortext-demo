import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import sklearn
import json

from nltk.tokenize.punkt import PunktSentenceTokenizer

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/ebm/"))
import quality3

class Handler():
    CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                    "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]

    title = "Risk of Bias"

    def load_models(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def __init__(self):
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "models/quality_models.pck"
        models_file = os.path.join(script_dir, rel_path)

        log.info("%s: loading models: %s" % (self.title, models_file))
        self.doc_models, self.doc_vecs, self.sent_models, self.sent_vecs = self.load_models(models_file)
        log.info("%s: done loading models" % (self.title))

        self.word_token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.sentence_tokenizer = PunktSentenceTokenizer()

    def handle(self, payload):
        """
        Adds sentence annotations and document predictions for all
        Risk of Bias core domains to the document as marginalia.
        """
        document = json.loads(payload)

        document_text = " ".join(document["pages"])

        sent_spans = self.sentence_tokenizer.span_tokenize(document_text)

        sent_text = [document_text[start:end] for start, end in sent_spans]

        output = []
        for test_domain, doc_model, doc_vec, sent_model, sent_vec in zip(self.CORE_DOMAINS, self.doc_models, self.doc_vecs, self.sent_models, self.sent_vecs):
            ####
            ## PART ONE - get the predicted sentences with risk of bias information
            ####
            annotations = []

            X_sents = sent_vec.transform(sent_text)
            pred_sents = [int(x_i) for x_i in sent_model.predict(X_sents)]

            positive_sents = [(index, sent) for index, (sent, pred) in enumerate(zip(sent_text, pred_sents)) if pred == 1]

            for index, sent in positive_sents:
                annotations += [{"content": sent, "uuid": str(uuid.uuid1()), "label": "biased"}]

            ####
            ## PART TWO - integrate summarized and full text, then predict the document class
            ####
            summary_text = " ".join([sent for index, sent in positive_sents])

            doc_vec.builder_clear()
            doc_vec.builder_add_docs([document_text])
            doc_vec.builder_add_docs([summary_text], prefix="high-prob-sent-")

            X_doc = doc_vec.builder_transform()

            document_prediction = "low" if doc_model.predict(X_doc)[0] == 1 else "uncertain"
            template_text = "**Overall risk of bias prediction**: %s"
            description = template_text % (document_prediction)

            output.append({
                "annotations": annotations,
                "description": description,
                "title": test_domain,
                "type": self.title
            })
        return json.dumps({"marginalia": output})
