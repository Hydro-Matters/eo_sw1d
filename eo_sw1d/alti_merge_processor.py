import geopandas as gpd
import glob
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from tqdm.autonotebook import tqdm
import zipfile

from .profiler import profiler

class AltiMergeProcessor:
    """ A class for merging multiple sources of altimetric data
    """

    def __init__(self, config: dict):
        """ Initialise an instance of AltiMergeProcessor

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

        self._logger.info("[START] AltiMergeProcessor::process") 

        # Load destination data
        if "_GDF" not in self.config["destination"]:
            if self.config["destination"]["source-type"] == "SWORD":
                dest_primaryKey = "node_id"
                merged_gdf = None
                for source_file in self.config["destination"]["source-files"]:
                    file_gdf = gpd.read_file(source_file)
                    if merged_gdf is None:
                        merged_gdf = file_gdf
                    else:
                        merged_gdf = pd.concat((merged_gdf, file_gdf), ignore_index=True)
                merged_gdf = merged_gdf.rename(columns={"wse": "p_wse", "wse_var": "p_wse_u", "width": "p_width", "width_var": "p_width_u"})
            self.config["destination"]["_GDF"] = merged_gdf
        destination_gdf = self.config["destination"]["_GDF"]
        self.crs = destination_gdf.crs

        if resume:

            # Load previously computed CSV file
            merged_csv_file = self.config["output"]["data-file"]
            if os.path.isfile(merged_csv_file):
                merged_df = pd.read_csv(merged_csv_file, sep=";", parse_dates=["t"])
                merged_data = {"key": [], "x": [], "y": [], "t": [], "source": [], "wse": [], "wse_u": [], "qual": [], "priority": []}
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
                    merged_data["wse"] += list(key_rows["wse"].values)
                    merged_data["wse_u"] += list(key_rows["wse_u"].values)
                    merged_data["qual"] += list(key_rows["qual"].values)
                    merged_data["priority"] += list(key_rows["priority"].values)
                    if "width" in merged_df.columns:
                        merged_data["width"] += list(key_rows["width"].values)
                self.merged_data = merged_data
                self.merged_df = merged_df

                # TODO check that merged_data keys correspond to destination_gdf keys

                return

        # Initialise dict of sources
        self.merged_alti = {}
        include_width = False
        for key in destination_gdf[dest_primaryKey].values:
            self.merged_alti[key] = {"t": [], "source": [], "wse": [], "wse_u": [], "qual": [], "priority": []}
            for source in self.config["sources"]:
                if source["source-type"] == "SWOT":
                    self.merged_alti[key]["width"] = []
                    include_width = True

        # Loop on sources
        priority_min = 9999
        priority_max = 0
        for index_source, source in enumerate(self.config["sources"]):

            self._logger.info("- Processing source:")
            self._logger.info("  - Type: %s" % source["source-type"])

            # Retrieve source priority index
            if "source-priority" in source:
                priority = source["source-priority"]
            else:
                priority = 1
            self._logger.info("  - Priority: %i" % priority)

            data_count = 0
            keys_found = []

            if source["source-type"] == "IceSat-2":

                # Find all IceSat-2 product files
                icesat2_files = glob.glob(os.path.join(source["source-dir"], "ATL*.h5"))
                self._logger.info("  - Number of files: %i" % len(icesat2_files))

                # Loop on IceSat-2 product files
                for file_index in tqdm(range(0, len(icesat2_files))):

                    # Process IceSat-2 product file
                    icesat2_file = icesat2_files[file_index]
                    icesat2_gdf = self.__process_IceSat2_file__(icesat2_file, destination_gdf, product=source["product"])

                    # Populate corresponding source's dict with loaded data
                    data_count += icesat2_gdf.index.size
                    for index in icesat2_gdf.index:
                        key = icesat2_gdf.loc[index, dest_primaryKey]
                        if not key in keys_found:
                            keys_found.append(key)

                        t = icesat2_gdf.loc[index, "date"]
                        wse = icesat2_gdf.loc[index, "wse"]
                        wse_u = icesat2_gdf.loc[index, "wse_u"]
                        self.merged_alti[key]["t"].append(t)
                        self.merged_alti[key]["source"].append(index_source+1)
                        self.merged_alti[key]["wse"].append(wse)
                        self.merged_alti[key]["wse_u"].append(wse_u)
                        if include_width:
                            self.merged_alti[key]["width"].append(np.nan)
                        self.merged_alti[key]["qual"].append(1)
                        self.merged_alti[key]["priority"].append(priority)
                        priority_min = min(priority_min, priority)
                        priority_max = max(priority_max, priority)


            elif source["source-type"] == "HYDROWEB":

                # Find all Hydroweb product files
                hydroweb_files = glob.glob(os.path.join(source["source-dir"], "hydroprd*.txt"))
                self._logger.info("  - Number of files: %i" % len(hydroweb_files))

                # Loop on Hydroweb product files
                for file_index in tqdm(range(0, len(hydroweb_files))):

                    # Process Hydroweb product file
                    hydroweb_file = hydroweb_files[file_index]
                    hydroweb_gdf = self.__process_Hydroweb_file__(hydroweb_file, destination_gdf)

                    # Populate corresponding source's dict with loaded data
                    data_count += hydroweb_gdf.index.size
                    for index in hydroweb_gdf.index:
                        key = hydroweb_gdf.loc[index, dest_primaryKey]
                        if not key in keys_found:
                            keys_found.append(key)
                        t = hydroweb_gdf.loc[index, "date"]
                        wse = hydroweb_gdf.loc[index, "wse"]
                        wse_u = hydroweb_gdf.loc[index, "wse_u"]
                        self.merged_alti[key]["t"].append(t)
                        self.merged_alti[key]["source"].append(index_source+1)
                        self.merged_alti[key]["wse"].append(wse)
                        self.merged_alti[key]["wse_u"].append(wse_u)
                        if include_width:
                            self.merged_alti[key]["width"].append(np.nan)
                        self.merged_alti[key]["qual"].append(1)
                        self.merged_alti[key]["priority"].append(priority)
                        priority_min = min(priority_min, priority)
                        priority_max = max(priority_max, priority)

                # TODO

            elif source["source-type"] == "SWOT":

                if "mapping" in source.keys():

                    # Load SWORD version-to-version mapping if prescribed in the configuration
                    mapping_file = source["mapping"]["file"]
                    node_mapping_csv = pd.read_csv(mapping_file, sep=",")
                    key_mapping = {}
                    for index in node_mapping_csv.index:
                        data_key = int(node_mapping_csv.loc[index, source["mapping"]["data_key"]])
                        dest_key = int(node_mapping_csv.loc[index, source["mapping"]["dest_key"]])
                        key_mapping[data_key] = dest_key
                else:
                   
                   key_mapping = None                
               
                if source["product"] == "RiverSP-Node":

                    # Find all SWOT RiverSP-Node product files
                    riversp_files = []
                    candidate_dirs = glob.glob(os.path.join(source["source-dir"], "SWOT_L2_HR_RiverSP_Node*"))
                    for candidate_dir in candidate_dirs:
                        if os.path.isdir(candidate_dir):
                            long_id = "_".join(os.path.basename(candidate_dir).split("_")[0:12])
                            if os.path.isfile(os.path.join(candidate_dir, "%s.shp" % long_id)):
                                riversp_files.append(os.path.join(candidate_dir, "%s.shp" % long_id))
                            elif os.path.isfile(os.path.join(candidate_dir, "%s.zip" % long_id)):
                                self._logger.info("- Extracting zip for %s" % long_id)
                                with zipfile.ZipFile(os.path.join(candidate_dir, "%s.zip" % long_id), 'r') as zp:
                                    zp.extractall(candidate_dir)
                                if not os.path.isfile(os.path.join(candidate_dir, "%s.shp" % long_id)):
                                    raise RuntimeError("shapefile not found after extracting the zip archive")
                                riversp_files.append(os.path.join(candidate_dir, "%s.shp" % long_id))
                            else:
                                self._logger.warning("No data (shp or zip) found in candidate directory: %s" % candidate_dir)
                    if len(riversp_files) == 0:
                        # Fallback for all riversp files in a single directory
                        riversp_files = glob.glob(os.path.join(source["source-dir"], "SWOT_L2_HR_RiverSP_Node*.shp"))
                    self._logger.info("  - Number of files: %i" % len(riversp_files))

                    # Process SWOT RiverSP-Node product files
                    for file_index in tqdm(range(0, len(riversp_files))):
                        riversp_file = riversp_files[file_index]


                        # Apply global filter and set data filtering parameters
                        crid = os.path.basename(riversp_file).split("_")[-2]
                        if "crids" in source:
                            if crid not in source["crids"]:
                                continue
                        if "max-node_q" in source:
                            max_node_q = source["max-node_q"]
                        else:
                            max_node_q = 99999999
                        if crid in source["mapping"]["crids"]:
                            node_mapping=key_mapping
                        else:
                            node_mapping=None
                        var_ext = None
                        if "var-ext" in source:
                            var_ext = source["var-ext"]
                        riversp_gdf = self.__process_SWOT_file__(riversp_file, destination_gdf, product=source["product"], max_node_q=max_node_q, bbox=destination_gdf.total_bounds, node_mapping=node_mapping, var_ext=var_ext)
                        data_count += riversp_gdf.index.size

                        # Populate corresponding source's dict with loaded data
                        for index in riversp_gdf.index:
                            key = int(riversp_gdf.loc[index, dest_primaryKey])
                            if not key in keys_found:
                                keys_found.append(key)
                            t = riversp_gdf.loc[index, "date"]
                            wse = riversp_gdf.loc[index, "wse"]
                            wse_u = riversp_gdf.loc[index, "wse_u"]
                            qual = riversp_gdf.loc[index, "qual"]
                            self.merged_alti[key]["t"].append(t)
                            self.merged_alti[key]["source"].append(index_source+1)
                            self.merged_alti[key]["wse"].append(wse)
                            self.merged_alti[key]["wse_u"].append(wse_u)
                            if not "width" in self.merged_alti[key]:
                                self.merged_alti[key]["width"] = []
                            width = riversp_gdf.loc[index, "width"]
                            self.merged_alti[key]["width"].append(width)
                            self.merged_alti[key]["qual"].append(qual)
                            self.merged_alti[key]["priority"].append(priority)
                            priority_min = min(priority_min, priority)
                            priority_max = max(priority_max, priority)

                else:

                    raise ValueError("Wrong SWOT product: %s" % source["product"])

            else:

                raise ValueError("Wrong source type: %s" % source["source-type"])
            
            self._logger.info("  - Number of nodes with data: %i" % len(keys_found))
            self._logger.info("  - Number of data: %i" % data_count)
            self._logger.info("  - Priority range: [%i - %i]" % (priority_min, priority_max))

        # Compute summary data
        summary_data = {"key": [], "x": [], "y": [], "wse_min": [], "wse_max": [], "obs_count": []}
        merged_data = {"key": [], "x": [], "y": [], "t": [], "source": [], "wse": [], "wse_u": [], "qual": [], "priority": []}
        for key in self.merged_alti.keys():
            point_gdf = destination_gdf[destination_gdf[dest_primaryKey] == key]
            x = point_gdf.loc[point_gdf.index[0], "x"]
            y = point_gdf.loc[point_gdf.index[0], "y"]

            # # Restrict using priority
            # priority_min_key = np.min(self.merged_alti[key]["priority"])
            # priority_values = np.array(self.merged_alti[key]["priority"])
            # priority_selection = priority_values[priority_values == priority_min_key]
            # t_selection = list(np.array(self.merged_alti[key]["t"])[priority_selection])
            # source_selection = list(np.array(self.merged_alti[key]["source"])[priority_selection])
            # wse_selection = list(np.array(self.merged_alti[key]["wse"])[priority_selection])
            # wse_u_selection = list(np.array(self.merged_alti[key]["wse_u"])[priority_selection])
            # qual_selection = list(np.array(self.merged_alti[key]["qual"])[priority_selection])
            merged_data["key"] += [key] * len(self.merged_alti[key]["t"])
            merged_data["x"] += [x] * len(self.merged_alti[key]["t"])
            merged_data["y"] += [y] * len(self.merged_alti[key]["t"])
            merged_data["t"] += self.merged_alti[key]["t"]
            merged_data["source"] += self.merged_alti[key]["source"]
            merged_data["wse"] += self.merged_alti[key]["wse"]
            merged_data["wse_u"] += self.merged_alti[key]["wse_u"]
            merged_data["qual"] += self.merged_alti[key]["qual"]
            merged_data["priority"] += self.merged_alti[key]["priority"]
            if "width" in self.merged_alti[key].keys():
                if not "width" in merged_data:
                    merged_data["width"] = []
                merged_data["width"] += self.merged_alti[key]["width"]
            summary_data["key"].append(key)
            summary_data["x"].append(x)
            summary_data["y"].append(y)
            if len(self.merged_alti[key]["wse"]) == 0:
                summary_data["wse_min"].append(np.nan)
                summary_data["wse_max"].append(np.nan)
                summary_data["obs_count"].append(0)
            else:
                summary_data["wse_min"].append(np.min(self.merged_alti[key]["wse"]))
                summary_data["wse_max"].append(np.max(self.merged_alti[key]["wse"]))
                summary_data["obs_count"].append(len(self.merged_alti[key]["wse"]))
            if "width" in self.merged_alti[key].keys():
                if not "width_min" in summary_data:
                    summary_data["width_min"] = []
                    summary_data["width_max"] = []
                if len(self.merged_alti[key]["width"]) == 0:
                    summary_data["width_min"].append(np.nan)
                    summary_data["width_max"].append(np.nan)
                else:
                    summary_data["width_min"].append(np.min(self.merged_alti[key]["width"]))
                    summary_data["width_max"].append(np.max(self.merged_alti[key]["width"]))

        merged_csv_file = self.config["output"]["data-file"]
        if "data-shapefile" in self.config["output"]:
            merged_data_shp_file = self.config["output"]["data-shapefile"]
            merge_data_shp = merged_data.copy()
            merge_data_shp = {key: value[:] for key, value in merge_data_shp.items()}
            merge_data_shp["t"] = [str(t) for t in merge_data_shp["t"]]
            merged_gdf = gpd.GeoDataFrame(data=merge_data_shp, geometry=gpd.points_from_xy(merged_data["x"], merged_data["y"]), crs=4326)
            merged_gdf.to_file(merged_data_shp_file)
        merged_df = pd.DataFrame(data=merged_data)
        merged_df.to_csv(merged_csv_file, sep=";", index=False)
        summy_shp_file = self.config["output"]["summary-shapefile"]
        summary_gdf = gpd.GeoDataFrame(data=summary_data, geometry=gpd.points_from_xy(summary_data["x"], summary_data["y"]), crs=4326)
        summary_gdf.to_file(summy_shp_file)

        self.merged_data = merged_data
        self.merged_df = merged_df

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

    def __process_IceSat2_file__(self, src, dest_gdf, product="ATL22", beams=None):
        """ Internal function that parse IceSat-2 product file

            Parameters
            ----------
                src: str
                    Path to the source (product) file
                dest_gdf: geopandas.GeoDataFrame
                    GeoDataFrame of the destination (objects where to extract data)
                product: str
                    Identifier of the IceSat-2 product contained in the source
                beams: list or None
                    List of beams to extracts. Extract all beams if set to None (default)

            Return
            ------
                merged_gdf: geopandas.GeoDataFrame
                    GeoDataFrame that contains all the extracted data
        """

        # Open product netCDF file
        dataset = nc.Dataset(src, "r")

        if beams is None:
            beams = ["1l", "1r", "2l", "2r", "3l", "3r"]

        if product == "ATL22":

            merged_gdf = None
            for beam in beams:
                beam_group = dataset.groups["gt%s" % beam]
                dates = list(beam_group.variables["transect_mean_time_utc"][:])
                dates = [date[0:19] for date in dates]
                lon = beam_group.variables["transect_mean_lon"][:]
                lat = beam_group.variables["transect_mean_lat"][:]
                wse = beam_group.variables["transect_mean_ht_ortho"][:]
                wse_u = beam_group.variables["transect_mean_stdev_water_surf"][:]
                qual = np.ones(len(wse))
                data_gdf = gpd.GeoDataFrame(data={"date": dates, "wse": wse, "wse_u": wse_u, "qual": qual}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
                joined_gdf = gpd.sjoin_nearest(dest_gdf.to_crs(3857), data_gdf.to_crs(3857), how="right", distance_col="dist", rsuffix="")
                joined_gdf = joined_gdf[joined_gdf["dist"] < joined_gdf["p_width"]]
                if merged_gdf is None:
                    merged_gdf = joined_gdf.loc[:, ["node_id", "wse", "wse_u", "date"]]
                else:
                    merged_gdf = pd.concat((merged_gdf, joined_gdf.loc[:, ["node_id", "wse", "wse_u", "date"]]), ignore_index=True, axis=0)
            if merged_gdf.index.size > 0:
                merged_gdf["date"] = pd.to_datetime(merged_gdf["date"])
            return merged_gdf

        else:

            raise NotImplementedError("Conversion for product %s is not implemented" % product)


    def __process_Hydroweb_file__(self, src, dest_gdf):
        """ Internal function that parse Hydroweb Nadir altimetry product file

            Parameters
            ----------
                src: str
                    Path to the source (product) file
                dest_gdf: geopandas.GeoDataFrame
                    GeoDataFrame of the destination (objects where to extract data)

            Return
            ------
                merged_gdf: geopandas.GeoDataFrame
                    GeoDataFrame that contains all the extracted data
        """

        # Read header
        ref_lon = None
        ref_lat = None
        with open(src, "r") as fp:
            line = fp.readline().strip()
            while line[0:1] == "#":
                words = line[1:].split("::")
                if len(words) > 1:
                    key = words[0].strip()
                    if key == "REFERENCE LONGITUDE":
                        ref_lon = float(words[1])
                    elif key == "REFERENCE LATITUDE":
                        ref_lat = float(words[1])
                line = fp.readline().strip()
        if ref_lon is None:
            raise RuntimeError("Reference longitude not found in file: %s" % src)
        if ref_lat is None:
            raise RuntimeError("Reference latitude not found in file: %s" % src)

        # Load data
        dataset = pd.read_csv(src, sep="\s+", comment="#", header=None)
        dataset['date'] = pd.to_datetime(dataset.pop(0)+' '+dataset.pop(1), format="%Y-%m-%d %H:%M")
        # dataset['date'] = dataset.pop(0)+' '+dataset.pop(1)
        # print(dataset)
        # choice = input()

        data = dataset.loc[:, ["date", 2, 3, 5, 6]]
        data = data.rename(columns={2: "wse", 3: "wse_u", 5: "lon", 6: "lat"})
        data_gdf = gpd.GeoDataFrame(data=data.loc[:, ["date", "wse", "wse_u"]], geometry=gpd.points_from_xy(data["lon"], data["lat"]), crs=4326)
        joined_gdf = gpd.sjoin_nearest(dest_gdf.to_crs(3857), data_gdf.to_crs(3857), how="right", distance_col="dist", rsuffix="")
        # print("JOINED")
        # print(joined_gdf)
        # print(np.nanmin(joined_gdf["dist"].values))
        # choice = input()
        joined_gdf = joined_gdf[joined_gdf["dist"] < joined_gdf["p_width"]]
        # if joined_gdf.index.size > 1:
        #     joined_gdf = joined_gdf.sort_values(by="dist", ascending=True)
        #     joined_gdf = joined_gdf.loc[joined_gdf.index[0:1], :]

        return joined_gdf

        # dataset = nc.Dataset(src, "r")
        # # print(src)

        # if beams is None:
        #     beams = ["1l", "1r", "2l", "2r", "3l", "3r"]

        # if product == "ATL22":

        #     merged_gdf = None
        #     for beam in beams:
        #         beam_group = dataset.groups["gt%s" % beam]
        #         dates = list(beam_group.variables["transect_mean_time_utc"][:])
        #         lon = beam_group.variables["transect_mean_lon"][:]
        #         lat = beam_group.variables["transect_mean_lat"][:]
        #         wse = beam_group.variables["transect_mean_ht_ortho"][:]
        #         wse_u = beam_group.variables["transect_mean_stdev_water_surf"][:]
        #         qual = np.ones(len(wse))
        #         data_gdf = gpd.GeoDataFrame(data={"date": dates, "wse": wse, "wse_u": wse_u, "qual": qual}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
        #         # print(data_gdf)
        #         # data_gdf.to_file("data.shp")
        #         joined_gdf = gpd.sjoin_nearest(dest_gdf.to_crs(3857), data_gdf.to_crs(3857), how="right", distance_col="dist", rsuffix="")
        #         # print(joined_gdf.loc[:, ["index_left", "wse", "wse_u"]])
        #         # choice = input()
        #         # print("[0]", joined_gdf.index.size, joined_gdf["dist"].min(), joined_gdf["dist"].max())
        #         # print("[*]", joined_gdf.sort_values(by="dist"))
        #         # joined_gdf.to_file("test.shp")
        #         joined_gdf = joined_gdf[joined_gdf["dist"] < joined_gdf["p_width"]]
        #         # print(joined_gdf.loc[:, ["index_left", "wse", "wse_u", "dist"]])
        #         # choice = input()
        #         # print("[1]", joined_gdf.index.size)
        #         # choice = input()
        #         # return joined_gdf.loc[:, ["node_id", "wse", "date"]]
        #         if merged_gdf is None:
        #             merged_gdf = joined_gdf.loc[:, ["node_id", "wse", "wse_u", "date"]]
        #         else:
        #             merged_gdf = pd.concat((merged_gdf, joined_gdf.loc[:, ["node_id", "wse", "wse_u", "date"]]), ignore_index=True, axis=0)
        #             # print("merged_gdf[1]", merged_gdf)
        #     return merged_gdf

        # else:

        #     raise NotImplementedError("Conversion for product %s is not implemented" % product)
               

    def __process_SWOT_file__(self, src, dest_gdf, product="RiverSP-Node", max_node_q=999999999, bbox=None, node_mapping=None, var_ext=None):

        if product == "RiverSP-Node":

            # profiler.start()
            riversp_gdf = gpd.read_file(src, engine="pyogrio", bbox=tuple(bbox))
            if var_ext is not None:
                column_mapping = {"wse" : "wse_0",
                                  "wse_u": "wse_u_0"}
                riversp_gdf = riversp_gdf.rename(columns=column_mapping)
                print("(0) columns:", riversp_gdf.columns)
                column_mapping = {"wse_%s" % var_ext : "wse",
                                  "wse_%s_u" % var_ext: "wse_u"}
                riversp_gdf = riversp_gdf.rename(columns=column_mapping)
                print("(1) columns:", riversp_gdf.columns)

            print("[0]: %i" % riversp_gdf.index.size)
            # profiler.step("read file")
            riversp_gdf["node_id"] = riversp_gdf["node_id"].astype(int)
            if node_mapping is not None:
                for index in riversp_gdf.index:
                    src_node_id = int(riversp_gdf.loc[index, "node_id"])
                    try:
                        dst_node_id = node_mapping[src_node_id]
                    except:
                        dst_node_id = -9999
                    riversp_gdf.loc[index, "node_id"] = dst_node_id
                riversp_gdf = riversp_gdf[riversp_gdf["node_id"] > -9999]

            # profiler.start()
            merged_gdf = pd.merge(dest_gdf, riversp_gdf, how="left", on="node_id", suffixes=["_left", ""])
            print("[1]: %i" % merged_gdf.index.size)
            # profiler.step("merge")
            # print(merged_gdf)
            # print(merged_gdf.columns)
            # rows = merged_gdf[merged_gdf["node_id"] == 23214300030171]
            # print("[0]", rows)
            merged_gdf = merged_gdf.loc[:, ["node_id", "wse", "wse_u", "width", "time_str", "node_q"]]
            merged_gdf = merged_gdf[np.logical_and(merged_gdf["wse"] > -9999, merged_gdf["time_str"] != "no_data")]
            merged_gdf["date"] = pd.to_datetime(merged_gdf["time_str"], format="%Y-%m-%dT%H:%M:%SZ")
            merged_gdf = merged_gdf[merged_gdf["wse"] > -9999]
            merged_gdf = merged_gdf[merged_gdf["node_q"] <= max_node_q]
            print("[2]: %i" % merged_gdf.index.size)
            # rows = merged_gdf[merged_gdf["node_id"] == 23214300030171]
            # print("[1]", rows)
            merged_gdf = merged_gdf.rename(columns={"node_q" : "qual"})
            return merged_gdf.loc[:, ["node_id", "wse", "wse_u", "width", "date", "qual"]]

        else:

            raise NotImplementedError("Conversion for product %s is not implemented" % product)

    

