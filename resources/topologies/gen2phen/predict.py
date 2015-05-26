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

from mintz_model import MintzModel
from annotate import annotate_text, get_concept

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

        logging.info("constructing model")
        self.model = MintzModel(script_dir);
        self.cls, self.vectorizer = self.load_model(os.path.join(script_dir, "models/model.pck"))

    def href(self, concept):
        return "[" + concept["label"] + "](" + concept["uri"] + ")"

    def predict(self, text):
        sents, abbrs, observed_entities = annotate_text(self.model, text)

        # map from entity_name -> concept_uri
        concept_map = {}

        # map from concept_uri -> definition
        concept_definitions = {}

        # map concept_uri -> sentence indexes
        sentence_indexes = defaultdict(set)

        # map of matched (gene, disease) -> sentence indicies
        matched_sentences = defaultdict(set)

        # map of disease_uri -> set(gene_uris)
        predicted_associations = defaultdict(set)

        def key_for(gene, disease):
            return gene + "-" + disease

        def annotation(sent_data):
            start = sent_data["bound"][0]
            stop = sent_data["bound"][1]

            return {"uuid": str(uuid.uuid1()),
                    "position": start,
                    "prefix": text[max(start - 32, 0):start],
                    "suffix": text[stop:min(stop + 32, len(text))],
                    "content": sent_data["sent"]}

        # BEGIN prediction loop over sentences with entities
        for idx, sent in enumerate(sents):
            sent_data, entities = sent

            for entity in entities:
                concept = get_concept(entity["kind"], entity["name"], entity.get("definition"))
                concept_map[entity["name"]] = concept["uri"]
                concept_definitions[concept["uri"]] = concept
                sentence_indexes[concept_uri].add(idx)

            included = sent_data["included"]
            excluded = sent_data["excluded"]
            pairs = self.model.get_pairs(entities)

            for pair in pairs:
                feature_dict = self.model.get_feature_dicts(included, excluded, pair)
                if not feature_dict:
                    continue
                X = self.vectorizer.transform(feature_dict)
                predict = self.cls.predict(X)
                if any(predict):

                    gene_uri = concept_map.get(pair[0]["name"])
                    disease_uri = concept_map.get(pair[1]["name"])

                    if gene_uri and disease_uri:
                        k = key_for(gene_uri, disease_uri)
                        matched_sentences[k].add(idx)
                        predicted_associations[disease_uri].add(gene_uri)
        # END predictions

        # BEGIN generate results
        output = []
        for disease_uri, gene_uris in predicted_associations:
            disease_concept = concept_map[disease_uri]
            gene_concepts = [concept_map[uri] for uri in gene_uris]

            disease_description = disease_concept.get("description", "*no description*")

            genes_list = "\n".join(["* " + self.href(g) for g in gene_concepts])

            description = "%s <br><br> %s" % (disease_description, gene_list)

            prediction = {"type": "gene-disease relation prediction",
                    "title": disease_concept["label"],
                    "description": description}

            annotations = []
            for gene_uri in gene_uris:
                k = key_for(gene_uri, disease_uri)
                indexes = matched_sentences[k]
                for idx in indexes:
                    sent_data, entities = sent_data[idx]
                    annotations.append(annotation(sent_data))

            prediction["annotations"] = annotations
            output.append(prediction)

        return output


    def handle(self, payload):
        document = json.loads(payload)

        text = " ".join(document["pages"])

        output = self.predict(fix_text(text))

        return json.dumps({"marginalia": output})
