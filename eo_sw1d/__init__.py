import logging

from .multisources_processor import MultiSourcesProcessor
from .mesh_processor import MeshProcessor
from .network_processor import NetworkProcessor

def initDefaultLogger():
    logger = logging.getLogger("eo-sw1d")
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter("[%(levelname)-5.5s]  %(message)s"))
    logger.addHandler(consoleHandler)
