import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import cPickle as pickle
import json

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/ebm/"))

from robotreviewer import biasrobot
from robotreviewer import PICO_robot

class Handler():
    def __init__(self):
        self.rob_bot = biasrobot.BiasRobot()
        self.pico_bot = PICO_robot.PICORobot()

    def handle(self, payload):
        """
        Adds sentence annotations and document predictions for all
        Risk of Bias core domains to the document as marginalia.
        """
        document = json.loads(payload)

        text = " ".join(document["pages"])

        annotations = self.rob_bot.annotate(text)
        PICO_annotations = self.pico_bot.annotate(text)

        annotations['marginalia'].extend(PICO_annotations['marginalia'])

        return json.dumps(annotations)
