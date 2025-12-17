import geopandas as gpd
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eo_sw1d.multisources_processor import MultiSourcesProcessor
from eo_sw1d.network_processor import NetworkProcessor
# from eo_sw1d.dassflow1d_mesh_builder import Dassflow1DMeshBuilder
# from eo_sw1d.observations_processor import ObservationsProcessor
# from eo_sw1d.hubeau_processor import HubEauProcessor
# from eo_sw1d.utils import HubEauProcessor

logger = logging.getLogger("eo-sw1d")
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter("[%(levelname)-5.5s]  %(message)s"))
logger.addHandler(consoleHandler)

config = {

    # MULTISOURCES PROCESSOR CONFIG
    "multisources-processor": {

        ## ALTIMETRIC DATA CONFIGURATION
        "alti-config": {
            "destination": {
                "source-type": "SWORD",
                "source-files": 
                [
                    "data/sword/nodes_subset_v17b.shp"
                ]
            },
            "sources": [
                {
                    "source-type": "HYDROWEB",
                    "source-priority": 1,
                    "source-dir": "data/hydroweb",
                },
                {
                    "source-type": "IceSat-2",
                    "source-priority": 1,
                    "source-dir": "data/icesat2/2019-2024",
                    "product": "ATL22"
                }
            ],
            "output": {
                "data-file": "output/garonne1_alti/alti_merged_v17b.csv",
                "data-shapefile": "output/garonne1_alti/alti_merged_v17b.shp",
                "summary-shapefile" : "output/garonne1_alti/summary_alti_merged_v17b.shp"
            }            
        },

        ## WIDTH DATA CONFIGURATION
        "width-config": {
            "sources": [
                {
                    "source-type": "BAS",
                    "source-priority": 1,
                    "source-dir": "data/bas",
                    "mapping": {
                        "file": "/mnt/c/WORK/DATA/SWORD/V17b/SWORD_v17b_to_v16_Translation/v17b/EU_NodeIDs_v17b_vs_v16.csv",
                        "file": "data/sword/subset_EU_NodeIDs_v17b_vs_v16.csv",
                        "data_key": "v16_node_id",
                        "dest_key": "v17_node_id"
                    }
                }
            ],
            "output": {
                "data-file": "output/garonne1_alti/test1_width_merged.shp",
                "summary-shapefile" : "output/garonne1_alti/test1_summary_width_merged.shp"
            }
            
        },

        "section-plots-prefix": "output/garonne1_alti/sections/section",
        "summary-shapefile" : "output/garonne1_alti/summary_alti_width_merged.shp"

    },

    # NETWORK PROCESSOR CONFIG
    "network-processor": {
        "source": {
            "source-type": "SWORD",
            "source-files":
            [
                "data/sword/reaches_subset_v17b.shp"
            ],
            "nodes-files":
            [
                "data/sword/nodes_subset_v17b.shp"
            ]
        },
        "output": {
            "shapefile": "output/garonne1_alti/network_v17b.shp",
            "segments-shapefile": "output/garonne1_alti/segments_v17b.shp"
        }

    }
}

do_process = False
if do_process is True:
    processor = MultiSourcesProcessor(config=config["multisources-processor"])
    merged_summary = processor.process(resume=(True, False))
    print("pause!");choice = input()

do_process = True
if do_process is True:
    network_processor = NetworkProcessor(config["network-processor"])
    network_processor.process(resume=False)
