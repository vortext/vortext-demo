import logging, copy, sys, uuid, os, re
log = logging.getLogger(__name__)

import json

sys.path.append('../../multilang/python')
sys.path.append(os.path.abspath("resources/topologies/ebm/"))
sys.path.append(os.path.abspath("resources/topologies/ebm/robotreviewer2"))

from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.ictrp_robot import ICTRPRobot


class Handler():
    def __init__(self):
        self.bias_bot = BiasRobot(top_k=3)
        self.pico_bot = PICORobot(top_k=1, min_k=1)
        self.ictrp_robot = ICTRPRobot(filter=["Date_registration", "Primary_outcome", "Secondary_outcome", "Target_size"])

    def handle(self, payload):
        """
        Adds sentence annotations and document predictions for all
        Risk of Bias core domains to the document as marginalia.
        """
        document = json.loads(payload.decode('utf-8'))
        text = " ".join(document["pages"])

        bots = [self.bias_bot, self.pico_bot, self.ictrp_robot]
        annotations = self.annotation_pipeline(bots, text)

        return json.dumps(annotations).encode('utf-8')

    def annotation_pipeline(self, bots, text):
        output = {"marginalia": []}
        for bot in bots:
            log.debug("Sending doc to {} for annotation...".format(bot.__class__.__name__))
            annotations = bot.annotate(text)
            output["marginalia"].extend(annotations["marginalia"])
            log.debug("{} done!".format(bot.__class__.__name__))
        return output
