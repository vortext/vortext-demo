from __future__ import generators
from noaho import NoAho

import sys, os, logging, csv, collections, functools, traceback, json, array
import cPickle as pickle
import os.path
from os import listdir, walk
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

import re
import itertools
from itertools import chain, repeat, islice
from ftfy import fix_text
import spacy.en

import numpy as np
import scipy as sp
import sklearn

import string

from fuzzywuzzy import process
from fuzzywuzzy import fuzz

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.grid_search import ParameterGrid

import nltk.tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction import DictVectorizer

from ftfy import fix_text

import ner

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def persist(file_name):
    """
    Persists the results of the function in a pickle
    To be used as a function decorator
    """

    script_dir = os.path.dirname(__file__)
    file_name_with_extension = os.path.join(script_dir, file_name + ".pck")
    def func_decorator(func):
        def func_wrapper(*args, **kwargs):
            if os.path.isfile(file_name_with_extension):
                logging.debug("loading...%s", file_name_with_extension)
                try:
                    f = open(file_name_with_extension, 'rb')
                    result = pickle.load(f)
                    f.close()
                    return result
                except Exception:
                    logging.warn(traceback.format_exc())
                    return None
            else:
                result = func(*args, **kwargs)
                try:
                    logging.debug("saving...%s", file_name_with_extension)
                    f = open(file_name_with_extension, 'wb')
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                except Exception:
                    logging.warn(traceback.format_exc())
                return result
        return func_wrapper
    return func_decorator


nlp = spacy.en.English()
from spacy.parts_of_speech import DET, NUM, PUNCT, X, PRT, NO_TAG, EOL

stupid = ["et", "as", "md", "mri", "pcr", "phd", "mlpa", "issn", "isbn", "dr", "drs", "mr", "mim", "all", "yes", "no"]
triplets = ["".join(triplet) for triplet in list(itertools.combinations_with_replacement(["a", "c", "g", "u"], 3))]
stop = set(stopwords.words('english')).union(stupid).union(triplets)

def is_eligable(tok):
    exclude = [NUM, X, NO_TAG, EOL]
    return tok.pos not in exclude

def tokenize(s, parse=False, tag=True):
    included = []
    excluded = []
    for tok in nlp(s, parse=parse, tag=tag):
        if is_eligable(tok):
            included.append(tok)
        else:
            excluded.append(tok)
    return included, excluded

def normalize(s):
    return fix_text(s.decode("utf-8", "ignore"))

def knuth_morris_pratt(text, pattern):
    # Knuth-Morris-Pratt string matching adapted for arbitrary sequences
    # David Eppstein, UC Irvine, 1 Mar 2002
    '''Yields all starting positions of copies of the pattern in the text.
    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    start_pos = 0
    match_len = 0
    for c in text:
        while match_len == len(pattern) or \
              match_len >= 0 and pattern[match_len] != c:
            start_pos += shifts[match_len]
            match_len -= shifts[match_len]
        match_len += 1
        if match_len == len(pattern):
            yield start_pos

def find_concepts(trie, words):
    s = " ".join([w for w in words if w.lower() not in stop])
    concepts = [s[k[0]:k[1]] for k in trie.findall_long(s) if (k[1] - k[0]) >= 2]
    # now only get things that were actual tokens (not parts of words)
    # ... if I were smarter&more patient I could probably modify the Aho-Corasick to do this in one go ...
    matches = []
    for concept in concepts:
        concept_words = [t.norm_ for t in nlp(concept)]
        matches.extend([(start,start + len(concept_words)) for start in knuth_morris_pratt(words, concept_words)])
    return matches

def find_concepts_ner(tagger, words):
    s = " ".join([w for w in words if w.lower() not in stop])
    result = tagger.get_entities(s)
    matches = defaultdict(list)
    for key, concepts in result.iteritems():
        for concept in concepts:
            concept_words = [t.norm_ for t in nlp(concept)]
            matches[key].extend([(start,start + len(concept_words)) for start in knuth_morris_pratt(words, concept_words)])
    return matches

def trie_from_file(file_name, transform=lambda x: x):
    with open(file_name, "r") as f:
        concepts = [transform(normalize(x.strip())) for x in f.readlines()]
    trie = NoAho()
    for concept in concepts:
        trie.add(concept)
        logging.debug("adding %s" % concept)
    return trie

def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)

def is_acronym(s):
    return s.isupper() and not any(c in string.whitespace for c in s)

def transform_abbr(s):
    return s.lower() if not is_acronym(s) else s

def str_fragment(tokens, pos):
    return " ".join([t.norm_ for t in tokens[pos[0]:pos[1]]])

class MintzModel:

    '''''
    These are lifted from
    `tmVar: A text mining approach for extracting sequence variants in biomedical literature` (Chih-Hsuan Wei et.al.)
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

    def __init__(self, data_dir):
        logging.debug("initializing")
        self.data_dir = data_dir
        self.sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
        self.hgvs_regexes = self.hgvs_regex.values()

        self.tagger = ner.SocketNER(host='localhost', port=9191)

        self.disease_trie = trie_from_file(os.path.join(self.data_dir, "concept_lists/diseases.txt"), transform=transform_abbr)
        self.gene_trie = trie_from_file(os.path.join(self.data_dir, "concept_lists/genes.txt"))

    def get_sentences(self, file_name):
        sents = []
        with open(file_name) as f:
            text = normalize(f.read())
            sents = self.sent_tokenize.tokenize(text, realign_boundaries=True)
        return [sent.rstrip(".") for sent in sents]

    def get_entities(self, words, use_ner=True):
        entities = defaultdict(set)

        entities["DISEASE"] = set(find_concepts(self.disease_trie, [transform_abbr(w) for w in words]))
        entities["GENE"] = set(find_concepts(self.gene_trie, words))

        if use_ner:
            entities_ner = find_concepts_ner(self.tagger, words)
            entities["DISEASE"] = entities["DISEASE"].union(entities_ner["DISEASE"])
            entities["GENE"] = entities["GENE"].union(entities_ner["GENE"])

        entities["GENE"] = entities["GENE"] - entities["DISEASE"]
        entities["DISEASE"] = entities["DISEASE"] - entities["GENE"]

        return entities

    def get_feature_dicts(self, sent, included, excluded, entity_pair):
        part_of_speech = [tok.norm_ + "/" + tok.pos_ for tok in included]

        entity = ((entity_pair[0], "GENE"),(entity_pair[1], "DISEASE"))

        pair = sorted(entity, key=lambda x: x[0])

        features = []
        start = pair[0][0][1]
        stop = pair[1][0][0]
        bound = sorted((start, stop))

        for window in range(0,3):
            feature_dict = {
                "first": pair[0][1],
                "second": pair[1][1]}

            feature_dict["pos"] = " ".join(part_of_speech[bound[0]:bound[1]])

            left_window = part_of_speech[max(0, start - window - 1):start]
            right_window = part_of_speech[stop + 1:min(len(part_of_speech) - 1, stop + window + 1)]

            feature_dict["left"] = " ".join(pad(left_window, window, "#PAD#"))
            feature_dict["right"] = " ".join(pad(right_window, window, "#PAD#"))

            features.append(feature_dict)
        return features

    @persist("data")
    def get_data(self, is_relevant, text_dirs):
        labels = []
        features = []

        for text_dir in text_dirs:
            for root, dir, files in os.walk(text_dir):
                n_files = len(files)
                for idx, name in enumerate(files):
                    pmid, ext = os.path.splitext(name)

                    logging.debug("processing %s %s (%s / %s)" % (text_dir, pmid, idx, n_files))
                    sentences_for_pmid = self.get_sentences(root + "/" + name)

                    for sent in sentences_for_pmid:
                        included, excluded = tokenize(sent)
                        entities = self.get_entities([tok.norm_ for tok in included])

                        for pair in itertools.product(entities["GENE"], entities["DISEASE"]):
                            gene_disease = (str_fragment(included, pair[0]), str_fragment(included, pair[1]))

                            if is_relevant(gene_disease):
                                feature_dicts = self.get_feature_dicts(sent, included, excluded, pair)
                                features.extend(feature_dicts)
                                labels.extend(repeat(True, len(feature_dicts)))
                            else:
                                feature_dicts = self.get_feature_dicts(sent, included, excluded, pair)
                                features.extend(feature_dicts)
                                labels.extend(repeat(False, len(feature_dicts)))

        return features, labels


    def get_relations(self, relations_file):
        with open(relations_file, 'rb') as f:
            relations = json.load(f)

        disease_gene_dict = defaultdict(set)
        gene_disease_dict = defaultdict(set)

        for relation in relations:
            disease_gene_dict[relation[0]].add(relation[1])
            gene_disease_dict[relation[1]].add(relation[0])

        return disease_gene_dict, gene_disease_dict


    def is_relevant_lambda(self, disease_gene_dict, gene_disease_dict):
        def is_relevant(pair, threshold=85):
            # a pair is (gene, disease) and will be checked against the ClinVar dicts
            diseases_for_gene = disease_gene_dict[pair[0]]
            return process.extractOne(pair[1], diseases_for_gene)[1] >= threshold if diseases_for_gene else False
        return is_relevant

    def evaluate(self):
        abstracts_dir = os.path.join(self.data_dir, "data/abstracts")
        full_text_dir = os.path.join(self.data_dir, "data/cache")

        relations_file = os.path.join(self.data_dir, "data/ClinVarFullRelease_2015-02-flattened.json")

        disease_gene_dict, gene_disease_dict = self.get_relations(relations_file)
        is_relevant = self.is_relevant_lambda(disease_gene_dict, gene_disease_dict)

        data, labels = self.get_data(is_relevant, [full_text_dir, abstracts_dir])
        v = DictVectorizer(sparse=True)
        X = v.fit_transform(data)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        clf = SGDClassifier(shuffle=True)

        params = {"alpha": [.00001, .0001, 0.01, 0.1, 1, 10],
                  "loss": ["log", "modified_huber"],
                  "penalty": ["l1", "l2", "elasticnet"]}
        search = GridSearchCV(clf, param_grid=params, scoring="f1", cv=5, refit=True, n_jobs=2)

        logging.debug("running predictions")
        search.fit(X_train, y_train)

        y_true, y_pred = y_test, search.predict(X_test)
        logging.info(classification_report(y_true, y_pred))

        logging.info(search.best_params_)

        with open(os.path.join(self.data_dir, "best_model.pck"), "wb") as f:
            pickle.dump([search.best_estimator_, v], f)

        return search.best_estimator_, v


def top100(vectorizer, clf):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    top = np.argsort(clf.coef_[0])[-100:]
    return [feature_names[j] for j in top]
