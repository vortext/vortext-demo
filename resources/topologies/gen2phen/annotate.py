import os, logging, csv, collections, functools, traceback,  array
log = logging.getLogger(__name__)

from fuzzywuzzy import process
from fuzzywuzzy import fuzz

import itertools
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib

from ftfy import fix_text
from abbreviations import get_abbreviations

from repoze.lru import lru_cache
import re


ESCAPE_CHARS_RE = re.compile(r'(?<!\\)(?P<char>[&|+\-!(){}[\]^\/"~*?:])')

def lucene_escape(value):
    r"""Escape un-escaped special characters and return escaped value.

    from https://fragmentsofcode.wordpress.com/2010/03/10/escape-special-characters-for-solrlucene-query/
    """
    return ESCAPE_CHARS_RE.sub(r'\\\\\g<char>', value)


def safeget(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


def get_sparql(query, endpoint="http://localhost:3030/gen2phen"):
    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        results = sparql.query().convert()
        return results["results"]
    except Exception, e:
        logging.error(e)
        return {}


def get_gene_concept(term):
    query  = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX text: <http://jena.apache.org/text#>
        PREFIX obo: <http://purl.obolibrary.org/obo/>


        SELECT DISTINCT ?concept ?label
        FROM  <http://localhost:3030/gen2phen/data/ogg>
        FROM  <http://localhost:3030/gen2phen/data/go>
        FROM  <http://localhost:3030/gen2phen/data/ordo>
        WHERE {
        SELECT ?concept ?label {
          ?concept text:query ("%(term)s" 1) ;
                   rdfs:label ?label} LIMIT 1}
    """ % {"term": lucene_escape(term)}

    r = get_sparql(query)
    if r and len(r["bindings"]) >= 1:
        bindings = r["bindings"][0]
        return {"uri": bindings["concept"]["value"],
                "label": bindings["label"]["value"]}
    else:
        return {}

def get_disease_concept(term):
    query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX obo: <http://purl.obolibrary.org/obo/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX efo: <http://www.ebi.ac.uk/efo/>
        PREFIX text: <http://jena.apache.org/text#>
        PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>
        PREFIX fn: <http://www.w3.org/2005/xpath-functions#>

        SELECT ?description ?concept ?label
        FROM  <http://localhost:3030/gen2phen/data/ordo>
        FROM  <http://localhost:3030/gen2phen/data/hp>
        FROM  <http://localhost:3030/gen2phen/data/doid>
        WHERE {
        SELECT ?description ?concept ?label {
          ?concept text:query ("%(term)s~" 1) ;
                   rdfs:label ?label .
          OPTIONAL {
          { ?concept obo:IAO_0000115 ?description }
            UNION { ?concept <http://www.ebi.ac.uk/efo/definition> ?description }
          }}} LIMIT 1
    """ % {"term": lucene_escape(term)}

    r = get_sparql(query)
    if r and len(r["bindings"]) >= 1:
        bindings = r["bindings"][0]

        return {"description": safeget(bindings, "description", "value") or "",
                "uri": bindings["concept"]["value"],
                "label": bindings["label"]["value"]}
    else:
        return {}

@lru_cache(maxsize=5000)
def get_concept(kind, term, definition=None):
    if kind == "DISEASE":
        return get_disease_concept(definition or term)
    elif kind == "GENE":
        return get_gene_concept(term)

def str_fragment(tokens, pos):
    return " ".join([t.norm_ for t in tokens[pos[0]:pos[1]]])

def annotate_text(model, text):
    # This is eehhm *academic* code. I.e. horribly written, terrible performance
    # It is intended to gather the needed info to make predictions based on sentences
    # *This should be destroyed after publication*

    # sentence tokenize the text
    sentence_bounds = model.get_sentences(text)
    sentence_texts = [text[sent[0]:sent[1]] for sent in sentence_bounds]

    # Given a list of sentences, find the abbreviations that are defined
    abbrs = get_abbreviations(sentence_texts)

    def is_defined(s):
        return s in abbrs

    def is_abbr(s):
        w = ''.join(c for c in s if c.isalnum())
        return w.isupper() or is_defined(s)

    keep_acronyms_for = set(["GENE"])

    # needed to correct the entity kind labels later on
    observed_entities = {}

    def observed_abbr_definition(term, threshold=90):
        # Fuzilly finds an observed abbreviation
        # This is a bit silly, but sometimes the abbreviation algorithm will come up with an
        # abbr that is not found by the NER. We attempt fuzzy matching as a last resort
        definition = abbrs.get(term)
        if definition in observed_entities:
            return definition
        else:
            fuzzy_match = process.extractOne(definition, observed_entities.keys())
            if not fuzzy_match:
                return None
            return fuzzy_match[0] if fuzzy_match[1] >= threshold else None

    results = list() # final results
    for idx, sent in enumerate(sentence_texts):
        # Tokenize the sentence, splitting into included and excluded tokens
        included, excluded = model.tokenize(sent)

        # For each of the sentences,
        # find the entities in the included tokens using NER and a ontology derived dictionary
        entities = model.get_entities([tok.norm_ for tok in included])

        # Loop over each of these (kind, [entity_intervals])
        # And build a data structure with relevant information
        predicted_entities = list()
        for kind, intervals in entities.iteritems():
            for interval in intervals:
                entity_name = str_fragment(included, interval)
                # Filter out all the irrelevant undefined abbreviations
                entity_is_abbr = is_abbr(entity_name)
                abbr_is_defined = entity_is_abbr and is_defined(entity_name)

                if not entity_is_abbr:
                    observed_entities[entity_name] = kind

                if kind in keep_acronyms_for \
                   or not entity_is_abbr \
                   or (entity_is_abbr and abbr_is_defined):
                    predicted_entities.append({"kind": kind,
                                               "name": entity_name,
                                               "interval": interval,
                                               "is_abbr": entity_is_abbr})
        sent_data = {"included": included,
                     "excluded": excluded,
                     "bound": sentence_bounds[idx],
                     "sent": sent}
        results.append((sent_data, predicted_entities))

    observed_abbrs = set()
    # Make the entity kinds congruent with their abbrs, if needed
    for sent_data, entities in results:
        for entity in entities:
            if entity["is_abbr"]:
                definition = observed_abbr_definition(entity["name"])
                if definition:
                    new_kind = observed_entities[definition]
                    entity["kind"] = new_kind
                    entity["definition"] = definition

                    observed_abbrs.add((entity["name"], new_kind, definition))

    # Lastly we insert all the abbreviations that were observed with a kind in the entities vector
    for sent_data, entities in results:
        words = [tok.norm_ for tok in sent_data["included"]]
        already_annotated = set([e["name"] for e in entities])
        for name, kind, definition in observed_abbrs:
            if name in already_annotated:
                continue

            indices = [i for i, w in enumerate(words) if w == name]
            for idx in indices:
                entities.append({"kind": kind,
                                 "name": name,
                                 "definition": definition,
                                 "is_implied": True,
                                 "is_abbr": True,
                                 "interval": (idx, idx + 1)})


    return results, abbrs, observed_entities
