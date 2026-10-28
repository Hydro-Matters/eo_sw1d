import geopandas as gpd
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eo_sw1d.multisources_processor import MultiSourcesProcessor
from eo_sw1d.network_processor import NetworkProcessor
from eo_sw1d.mesh_processor import MeshProcessor
# from eo_sw1d.observations_processor import ObservationsProcessor
# from eo_sw1d.hubeau_processor import HubEauProcessor
# from eo_sw1d.utils import HubEauProcessor

import warnings
warnings.filterwarnings(action="error")

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
            # "segments-shapefile": "output/garonne1_alti/segments_v17b.shp"
        }

    },

    # MESH PROCESSOR
    "mesh-processor": {

        "sources": {
            "network-file": "output/garonne1_alti/network_v17b.shp",
            "nodes-file": "data/sword/nodes_subset_v17b.shp",
            "recompute-nodes-distances": True,
            "profiles-file": "output/garonne1_alti/summary_alti_width_merged.shp"
        },
        "inflow": [
            {
                "segment-index": 0,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O4001010_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 1,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O4802520_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 2,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O1652930_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 3,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O1252510_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 4,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O0200020_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 5,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O0502520_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 7,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O5602520_Qln(n=15_non-glissant).csv"
            },
            {
                "segment-index": 16,
                "hydro3-file": "/mnt/c/WORK/PROJETS/HYDRAVATAR/Garonne/Modeles/data/gauges/SCHAPI/O7701540_Qln(n=15_non-glissant).csv"
            },
        ],
        "output": {
            "mesh-file-prefix": "output/garonne1_alti/mesh_v17b01",
            "mesh-shapefile": "output/garonne1_alti/mesh_v17b01.shp",
            "plot-dir": "output/garonne1_alti/plot",
            "tmp-dir": "output/garonne1_alti/tmp",
        }
    }
}

processor = MultiSourcesProcessor(config=config["multisources-processor"])
merged_summary = processor.process(resume=(True, True))

network_processor = NetworkProcessor(config["network-processor"])
network_processor.process(resume=False)

mesh_processor = MeshProcessor(config["mesh-processor"])
mesh_processor.process(resume=False)
