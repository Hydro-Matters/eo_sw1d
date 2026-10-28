import geopandas as gpd
import glob
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from tqdm.autonotebook import tqdm
import zipfile

class WidthMergeProcessor:
    """ A class for merging multiple sources of width data
    """

    def __init__(self, config):
        """ Initialise an instance of WidthMergeProcessor

            Parameters
                config: dict
                    Configuration
        """
        # Retrieve loggger
        self._logger = logging.getLogger("eo-sw1d")

        # Store configuration
        self.config = config


    def process(self, resume: bool=False):
        """Run the main process

            Parameters
                resume: bool
                    True to resume from a previous run
        """

        self._logger.info("[START] WidthMergeProcessor::process") 

        # Load destination data
        if self.config["destination"]["source-type"] == "SWORD":
            dest_primaryKey = "node_id"
        if "_GDF" not in self.config["destination"]:
            if self.config["destination"]["source-type"] == "SWORD":
                merged_gdf = None
                for source_file in self.config["destination"]["source-files"]:
                    file_gdf = gpd.read_file(source_file)
                    if merged_gdf is None:
                        merged_gdf = file_gdf
                    else:
                        merged_gdf = pd.concat((merged_gdf, file_gdf), ignore_index=True)
            self.config["destination"]["_GDF"] = merged_gdf
        destination_gdf = self.config["destination"]["_GDF"]

        if resume:
            merged_csv_file = self.config["output"]["data-file"]
            # print("merged_csv_file: %s" % merged_csv_file)
            # choice = input()
            if os.path.isfile(merged_csv_file):
                merged_df = pd.read_csv(merged_csv_file, sep=";", parse_dates=["t"])
                merged_data = {"key": [], "x": [], "y": [], "t": [], "source":[], "width": [], "priority": []}
                if "width" in merged_df.columns:
                    merged_data["width"] = []
                if "priority" not in merged_df.columns:
                    merged_df["priority"] = np.ones(merged_df.index.size, dtype=int)
                if "source" not in merged_df.columns:
                    merged_df["source"] = np.ones(merged_df.index.size, dtype=int)
                for key in merged_df["key"].unique():
                    key_rows = merged_df[merged_df["key"] == key]
                    if key_rows.index.size == 0:
                        raise RuntimeError("No data for key: %i" % key)
                    x = key_rows.loc[key_rows.index[0], "x"]
                    y = key_rows.loc[key_rows.index[0], "y"]
                    merged_data["key"] += [key] * key_rows.index.size
                    merged_data["x"] += [x] * key_rows.index.size
                    merged_data["y"] += [y] * key_rows.index.size
                    merged_data["t"] += list(key_rows["t"].values)
                    merged_data["source"] += list(key_rows["source"].values)
                    merged_data["width"] += list(key_rows["width"].values)
                    merged_data["priority"] += list(key_rows["priority"].values)
                self.merged_data = merged_data
                # merged_df = pd.DataFrame(data=merged_data)
                self.merged_df = merged_df

                # TODO check that merged_data keys correspond to destination_gdf keys
                # print("HERE:", self.merged_data)

                return

        self.merged_width = {}
        for key in destination_gdf[dest_primaryKey].values:
            self.merged_width[key] = {"t": [], "source": [], "width": [], "priority": []}

        for index_source, source in enumerate(self.config["sources"]):

            print("- Processing source:")
            print("  - Type: %s" % source["source-type"])

            if "source-priority" in source:
                priority = source["source-priority"]
            else:
                priority = 1
            print("  - Priority: %i" % priority)

            if "mapping" in source.keys():
                mapping_file = source["mapping"]["file"]
                node_mapping_csv = pd.read_csv(mapping_file, sep=",")
                key_mapping = {}
                # inverse_key_mapping = {}
                for index in node_mapping_csv.index:
                    data_key = int(node_mapping_csv.loc[index, source["mapping"]["data_key"]])
                    dest_key = int(node_mapping_csv.loc[index, source["mapping"]["dest_key"]])
                    # if data_key == 61660100080015:
                    #     print("FOUND: data=%i, dest=%i" % (data_key, dest_key))
                    #     choice = input()
                    key_mapping[data_key] = dest_key

            if source["source-type"] == "BAS":

                # Find all tiles files
                bas_csv_files = []
                bas_tiles = glob.glob(os.path.join(source["source-dir"], "T*"))
                for bas_tile in bas_tiles:
                   
                    if os.path.isdir(bas_tile):
                   
                        datedirs = glob.glob(os.path.join(bas_tile, "20*"))

                        for datedir in datedirs:

                            date_csv_files = glob.glob(os.path.join(datedir, "*csv"))
                            if len(date_csv_files) > 1:
                                raise RuntimeError("Multiple csv files in directory: %s" % datedir)
                            elif len(date_csv_files) == 0:
                                print("[WARNING] No csv files in directory: %s" % datedir)
                            else:
                                bas_csv_files.append(date_csv_files[0])

                print("BAS CSV files: %i" % len(bas_csv_files))

                for file_index in tqdm(range(0, len(bas_csv_files))):
                    bas_csv_file = bas_csv_files[file_index]
                    bas_gdf = self.__process_BAS_file__(bas_csv_file, destination_gdf, node_mapping=key_mapping)
                    for index in bas_gdf.index:
                        key = bas_gdf.loc[index, dest_primaryKey]
                        if key in self.merged_width.keys():
                            t = bas_gdf.loc[index, "datetime"]
                            width = bas_gdf.loc[index, "width"]
                            self.merged_width[key]["t"].append(t)
                            self.merged_width[key]["source"].append(index_source)
                            self.merged_width[key]["width"].append(width)
                            self.merged_width[key]["priority"].append(priority)

            elif source["source-type"] == "BAS-old":

                # Find all tiles files
                bas_csv_files = glob.glob(os.path.join(source["source-dir"], "RESULT*.csv"))

                print("BAS(old) CSV files: %i" % len(bas_csv_files))

                for file_index in tqdm(range(0, len(bas_csv_files))):
                    bas_csv_file = bas_csv_files[file_index]
                    bas_gdf = self.__process_BASold_file__(bas_csv_file, destination_gdf, node_mapping=key_mapping)
                    for index in bas_gdf.index:
                        key = bas_gdf.loc[index, dest_primaryKey]
                        if key in self.merged_width.keys():
                            t = bas_gdf.loc[index, "datetime"]
                            width = bas_gdf.loc[index, "width"]
                            self.merged_width[key]["t"].append(t)
                            self.merged_width[key]["source"].append(index_source)
                            self.merged_width[key]["width"].append(width)
                            self.merged_width[key]["priority"].append(priority)

            # elif source["source-type"] == "SWOT":
               
            #     if source["product"] == "RiverSP-Node":

            #     # Find all data files
            #     riversp_files = []
            #     candidate_dirs = glob.glob(os.path.join(source["source-dir"], "SWOT_L2_HR_RiverSP_Node*"))
            #     for candidate_dir in candidate_dirs:
            #         if os.path.isdir(candidate_dir):
            #             long_id = "_".join(os.path.basename(candidate_dir).split("_")[0:12])
            #             if os.path.isfile(os.path.join(candidate_dir, "%s.shp" % long_id)):
            #                 riversp_files.append(os.path.join(candidate_dir, "%s.shp" % long_id))
            #             elif os.path.isfile(os.path.join(candidate_dir, "%s.zip" % long_id)):
            #                 print("- Extracting zip for %s" % long_id)
            #                 with zipfile.ZipFile(os.path.join(candidate_dir, "%s.zip" % long_id), 'r') as zp:
            #                     zp.extractall(candidate_dir)
            #                 if not os.path.isfile(os.path.join(candidate_dir, "%s.shp" % long_id)):
            #                     raise RuntimeError("shapefile not found after extracting the zip archive")
            #             else:
            #                 print("WARNING no data (shp or zip) found in candidate directory: %s" % candidate_dir)


            #    for index in tqdm(range(0, len(riversp_files))):
            #        riversp_file = riversp_files[index]
            #        riversp_gdf = self.__process_SWOT_file__(riversp_file, destination_gdf, product=source["product"])
            #     #    if 
            #     #    print(icesat2_gdf.loc[:, [dest_primaryKey, "date", "wse_right"]])
            #     #    choice = input()
            #        for index in riversp_gdf.index:
            #            key = riversp_gdf.loc[index, dest_primaryKey]
            #            t = riversp_gdf.loc[index, "date"]
            #            wse = riversp_gdf.loc[index, "wse"]
            #            self.merged_width[key]["t"].append(t)
            #            self.merged_width[key]["wse"].append(wse)


                #    merged_width
        summary_data = {"key": [], "x": [], "y": [], "width_min": [], "width_max": [], "obs_count": []}
        merged_data = {"key": [], "x": [], "y": [], "t": [], "source":[], "width": [], "priority": []}
        for key in self.merged_width.keys():
            point_gdf = destination_gdf[destination_gdf[dest_primaryKey] == key]
            x = point_gdf.loc[point_gdf.index[0], "x"]
            y = point_gdf.loc[point_gdf.index[0], "y"]
            merged_data["key"] += [key] * len(self.merged_width[key]["t"])
            merged_data["x"] += [x] * len(self.merged_width[key]["t"])
            merged_data["y"] += [y] * len(self.merged_width[key]["t"])
            merged_data["t"] += self.merged_width[key]["t"]
            merged_data["source"] += self.merged_width[key]["source"]
            merged_data["width"] += self.merged_width[key]["width"]
            merged_data["priority"] += self.merged_width[key]["priority"]
            summary_data["key"].append(key)
            summary_data["x"].append(x)
            summary_data["y"].append(y)
            if len(self.merged_width[key]["width"]) == 0:
                summary_data["width_min"].append(np.nan)
                summary_data["width_max"].append(np.nan)
            else:
                summary_data["width_min"].append(np.min(self.merged_width[key]["width"]))
                summary_data["width_max"].append(np.max(self.merged_width[key]["width"]))
            summary_data["obs_count"].append(len(self.merged_width[key]["width"]))
        merged_csv_file = self.config["output"]["data-file"]
        merged_df = pd.DataFrame(data=merged_data)
        merged_df.to_csv(merged_csv_file, sep=";", index=False)
        summy_shp_file = self.config["output"]["summary-shapefile"]
        summary_gdf = gpd.GeoDataFrame(data=summary_data, geometry=gpd.points_from_xy(summary_data["x"], summary_data["y"]), crs=4326)
        summary_gdf.to_file(summy_shp_file)

        self.merged_data = merged_data
        self.merged_df = merged_df

        self._logger.info("[ END ] WidthMergeProcessor::process") 

    def get_data(self, key=None):
        """ Retrieve data for all sources or for a single source

            Parameters
            ----------
                key: str
                    Source identifier
        """

        if key is None:
            return self.merged_df
        else:
            return self.merged_df[self.merged_df["key"] == key]

    def __process_BAS_file__(self, src, dest_gdf, node_mapping=None):
        """ Internal function that parse BAS product file

            Parameters
            ----------
                src: str
                    Path to the source (product) file
                dest_gdf: geopandas.GeoDataFrame
                    GeoDataFrame of the destination (objects where to extract data)
                node_mapping: dict
                    Node mapping (from different versions of SWORD for instance)

            Return
            ------
                loaded_df: geopandas.GeoDataFrame
                    DataFrame that contains all the loaded data
        """

        dataset = pd.read_csv(src, sep=",", parse_dates=["datetime"])
        # dataset = dataset.rename(columns={"datetime": "t"})
        if "vs_id" in dataset.columns:
            dataset = dataset.rename(columns={"vs_id": "node_id"})
        dataset["node_id"] = dataset["node_id"].astype(int)

        if node_mapping is not None:
            for index in dataset.index:
                src_node_id = int(dataset.loc[index, "node_id"])
                try:
                    dst_node_id = node_mapping[src_node_id]
                except:
                    dst_node_id = -9999
                dataset.loc[index, "node_id"] = dst_node_id
            dataset = dataset[dataset["node_id"] > -9999]

        loaded_df = dataset.loc[:, ["node_id", "width", "datetime"]]

        return loaded_df

    # DEPRECATED
    def __process_BASold_file__(self, src, dest_gdf, node_mapping=None):
        """ Internal function that parse BAS product file in old format

            Parameters
            ----------
                src: str
                    Path to the source (product) file
                dest_gdf: geopandas.GeoDataFrame
                    GeoDataFrame of the destination (objects where to extract data)
                node_mapping: dict
                    Node mapping (from different versions of SWORD for instance)

            Return
            ------
                loaded_df: geopandas.GeoDataFrame
                    DataFrame that contains all the loaded data
        """

        dataset = pd.read_csv(src, sep=",")
        date_str = os.path.basename(src).split("_")[1]
        datetime = np.datetime64("%s-%s-%s" % (date_str[0:4], date_str[4:6], date_str[6:]))
        # dataset = dataset.rename(columns={"datetime": "t"})
        if "vs_id" in dataset.columns:
            dataset = dataset.rename(columns={"vs_id": "node_id"})
        dataset["node_id"] = dataset["node_id"].astype(int)
        dataset["datetime"] = datetime

        if node_mapping is not None:
            for index in dataset.index:
                src_node_id = int(dataset.loc[index, "node_id"])
                try:
                    dst_node_id = node_mapping[src_node_id]
                except:
                    dst_node_id = -9999
                dataset.loc[index, "node_id"] = dst_node_id
            dataset = dataset[dataset["node_id"] > -9999]

        loaded_df = dataset.loc[:, ["node_id", "width", "datetime"]]

        return loaded_df
