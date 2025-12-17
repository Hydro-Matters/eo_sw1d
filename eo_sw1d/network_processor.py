import geopandas as gpd
import glob
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from shapely.geometry import LineString
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

class NetworkProcessor:
    """ A class for computing river network
    """

    def __init__(self, config):
        """ Initialise an instance of NetworkProcessor

            Parameters
                config: dict
                    Configuration
        """

        self._logger = logging.getLogger("eo-sw1d")
        self.config = config

    def process(self, resume: bool=False, segments_mapping: dict=None):
        """ Run the main process

            Parameters
                resume: bool
                    True to resume from a previous run
                segments_mapping: dict
                    True to resume from a previous run
        """

        self._logger.info("[START] NetworkProcessor::process") 

        # Load source files
        self._logger.info("Load sources files")
        if self.config["source"]["source-type"] == "SWORD":
            merged_gdf = None
            for source_file in self.config["source"]["source-files"]:
                file_gdf = gpd.read_file(source_file)
                if merged_gdf is None:
                    merged_gdf = file_gdf
                else:
                    merged_gdf = pd.concat((merged_gdf, file_gdf), ignore_index=True)
            merged_gdf = merged_gdf.rename(columns={"wse": "p_wse", "wse_var": "p_wse_u", "width": "p_width", "width_var": "p_width_u"})
        else:
            raise ValueError("Wrong source type: %s" % self.config["source"]["source-type"])
        
        # Load node files
        if "nodes-files" in self.config["source"]:
            merged_nodes_gdf = None
            for source_file in self.config["source"]["nodes-files"]:
                file_gdf = gpd.read_file(source_file)
                if merged_nodes_gdf is None:
                    merged_nodes_gdf = file_gdf
                else:
                    merged_nodes_gdf = pd.concat((merged_nodes_gdf, file_gdf), ignore_index=True)
            merged_nodes_gdf = merged_nodes_gdf.rename(columns={"wse": "p_wse", "wse_var": "p_wse_u", "width": "p_width", "width_var": "p_width_u"})
        else:
            merged_nodes_gdf = None

        # Initialize seg_idx (segment index) and c_dist_out (corrected dist out)
        merged_gdf["seg_idx"] = np.ones(merged_gdf.index.size, dtype=int) * -9
        merged_gdf["c_dist_out"] = np.ones(merged_gdf.index.size) * -9999.0

        self._logger.info("Compute segments:")

        # Retrieve downstream reaches
        downstream_rows = merged_gdf[merged_gdf["n_rch_dn"] == 0]
        downstream_rows.sort_values(by="facc")
        merged_gdf.loc[downstream_rows.index, "c_dist_out"] = merged_gdf.loc[downstream_rows.index, "dist_out"]
        self._logger.info("- Downstream reaches: [%s]" % ",".join(["%i" % downstream_rows.loc[index, "reach_id"] for index in downstream_rows.index]))

        # Build network from downstream to upstream
        segment_index = -1
        stack = list(downstream_rows.index)
        while len(stack) > 0:

            # Get next element
            cur_row_index = stack.pop(0)

            # Test if current element has no segment index
            if merged_gdf.loc[cur_row_index, "seg_idx"] > -9:
                continue

            # Increase segment index
            segment_index += 1

            # Traverse segment until a junction or most upstream is reached
            while True:

                # Set segment index to current element
                merged_gdf.loc[cur_row_index, "seg_idx"] = segment_index

                # Compute c_dist_out
                dn_dist_out_max = -9999.0
                rch_id_dn = merged_gdf.loc[cur_row_index, "rch_id_dn"]
                if rch_id_dn is None:
                    merged_gdf.loc[cur_row_index, "c_dist_out"] = merged_gdf.loc[cur_row_index, "dist_out"]
                else:
                    rch_id_dn = list(map(int, rch_id_dn.split(" ")))

                    for i in range(0, len(rch_id_dn)):
                        reach_rows = merged_gdf[merged_gdf["reach_id"] == rch_id_dn[i]]
                        dn_dist_out_max = max(dn_dist_out_max, merged_gdf.loc[reach_rows.index[0], "c_dist_out"])

                    merged_gdf.loc[cur_row_index, "c_dist_out"] = dn_dist_out_max + merged_gdf.loc[cur_row_index, "reach_len"]

                # Test end conditions for current segment
                if merged_gdf.loc[cur_row_index, "n_rch_up"] == 0:
                    
                    # Most upstream has been reached (no upstream element)
                    break

                elif merged_gdf.loc[cur_row_index, "n_rch_up"] > 1:

                    # Add next reaches in stack for a defluence
                    rch_id_up = merged_gdf.loc[cur_row_index, "rch_id_up"].split(" ")
                    if len(rch_id_up) != merged_gdf.loc[cur_row_index, "n_rch_up"]:
                        raise RuntimeError("Number of upstream reachIDs and n_rch_up do not agree")
                    for reach_id in rch_id_up:
                        reach_id = int(reach_id)
                        reach_rows = merged_gdf[merged_gdf["reach_id"] == reach_id]
                        if reach_rows.index.size == 0:
                            raise RuntimeError("reach_id not found: %i" % reach_id)
                        elif reach_rows.index.size > 1:
                            print(reach_rows)
                            raise RuntimeError("Found multiple rows for reach_id: %i" % reach_id)
                        stack.append(reach_rows.index[0])
                    break

                else:

                    reach_id = int(merged_gdf.loc[cur_row_index, "rch_id_up"])
                    reach_rows = merged_gdf[merged_gdf["reach_id"] == reach_id]
                    if reach_rows.index.size == 0:
                        break
                    elif reach_rows.index.size > 1:
                        raise RuntimeError("Found multiple rows for reach_id: %i" % reach_id)
                    cur_row_index = reach_rows.index[0]
                    if merged_gdf.loc[cur_row_index, "n_rch_dn"] > 1:
                        stack.append(cur_row_index)
                        break

        self._logger.info("- Number of segments: %i" % segment_index)

        # TO COMMENT ?
        #merged_gdf["seg_idx"] = segment_index - merged_gdf["seg_idx"]

        # Apply user mapping
        if segments_mapping is not None:
            for index in merged_gdf.index:
                seg_idx = merged_gdf.loc[index, "seg_idx"]
                if seg_idx in segments_mapping:
                    merged_gdf.loc[index, "seg_idx"] = segments_mapping[seg_idx]

        # Save to shapefile
        merged_gdf = merged_gdf[merged_gdf["seg_idx"] != -9]
        merged_gdf.to_file(self.config["output"]["shapefile"])
        self._merged_gdf = merged_gdf

        # Save segments to shapefile
        if "segments-shapefile" in self.config["output"]:

            # Compute aggregated segments
            segments_indices = merged_gdf["seg_idx"].unique()

            segments_data = {"seg_idx": [],
                             "geometry": []}

            for seg_idx in segments_indices:

                segment_gdf = merged_gdf[merged_gdf["seg_idx"] == seg_idx]
                segment_gdf_sorted = segment_gdf.sort_values(by="dist_out", ascending=False)
                coords = []
                for index in segment_gdf_sorted.index:
                    reach_coords = segment_gdf_sorted.loc[index, "geometry"].coords
                    reach_id = segment_gdf_sorted.loc[index, "reach_id"]
                    if merged_nodes_gdf is not None:
                        reach_nodes_gdf = merged_nodes_gdf[merged_nodes_gdf["reach_id"] == reach_id]
                        reach_nodes_gdf = reach_nodes_gdf.sort_values(by="dist_out")
                        n0 = reach_nodes_gdf["geometry"].values[0]
                        nE = reach_nodes_gdf["geometry"].values[-1]
                        # x0 = reach_coords[0][0]
                        # y0 = reach_coords[0][1]
                        # xE = reach_coords[-1][0]
                        # yE = reach_coords[-1][1]
                        d1 = segment_gdf_sorted.loc[index, "geometry"].project(n0)
                        d2 = segment_gdf_sorted.loc[index, "geometry"].project(nE)
                        # l = ax0.plot([c[0] for c in reach_coords[::-1]], [c[1] for c in reach_coords[::-1]], "-", lw=4, label="%i" % reach_id)
                        if d1 < d2:
                            # print("%i: std" % reach_id)
                            # print("Upstream: (%f, %f)" % (reach_coords[-1][0], reach_coords[-1][1]))
                            # print("Dowstream: (%f, %f)" % (reach_coords[1][0], reach_coords[1][1]))
                            # print("Dowstream++: (%f, %f)" % (reach_coords[0][0], reach_coords[0][1]))
                            coords += reach_coords[::-1][1:]
                            # ax0.plot(reach_coords[-1][0], reach_coords[-1][1], "o", color=l[0].get_color())
                            # ax0.plot(reach_coords[0][0], reach_coords[0][1], "+", color=l[0].get_color())
                        else:
                            # print("%i: reverse" % reach_id)
                            # print("Upstream+: (%f, %f)" % (reach_coords[0][0], reach_coords[0][1]))
                            # print("Upstream: (%f, %f)" % (reach_coords[1][0], reach_coords[1][1]))
                            # print("Dowstream: (%f, %f)" % (reach_coords[-1][0], reach_coords[-1][1]))
                            # print("coords[0]:", coords[-2:])
                            coords += reach_coords[1:]
                            # print("coords[1]:", coords[-2:])
                            # ax0.plot(reach_coords[0][0], reach_coords[0][1], "o", color=l[0].get_color())
                            # ax0.plot(reach_coords[-1][0], reach_coords[-1][1], "+", color=l[0].get_color())

                # ax0.set_title("Segment")
                # ax0.plot([c[0] for c in coords], [c[1] for c in coords], "k--")
                # ax0.legend()
                # plt.show()

                segments_data["seg_idx"].append(seg_idx)
                segments_data["geometry"].append(LineString(coords))

            segments_gdf = gpd.GeoDataFrame(data={"seg_idx": segments_data["seg_idx"]}, geometry=segments_data["geometry"], crs=4326)
            segments_gdf.to_file(self.config["output"]["segments-shapefile"])
