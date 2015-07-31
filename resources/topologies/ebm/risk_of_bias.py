import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import json

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/ebm/"))

from robotreviewer import biasrobot

class Handler():
    def __init__(self):
        self.bot = biasrobot.BiasRobot()

    def handle(self, payload):
        """
        Adds sentence annotations and document predictions for all
        Risk of Bias core domains to the document as marginalia.
        """
        document = json.loads(payload)

        text = " ".join(document["pages"])

        return json.dumps(self.bot.annotate(text))
