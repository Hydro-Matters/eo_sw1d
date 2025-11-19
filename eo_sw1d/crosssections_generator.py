import geopandas as gpd
from geopy.distance import distance
import glob
import logging
import netCDF4 as nc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm.autonotebook import tqdm
from shapely.geometry import LineString
import zipfile

from .profiler import profiler

def calc_bearing(lat1, long1, lat2, long2) -> float:
    """ Helper function that calculate the bearing from two points coordinates

        Parameters
        ----------
            lat1 : float
                Latitude of origin point
            lon1 : float
                Longitude of origin point
            lat2 : float
                Latitude of destination point
            lon2 : float
                Longitude of destination point

        Retrun
            bearing: float
                Bearing in degrees
    """
    # Convert latitude and longitude to radians
    lat1 = math.radians(lat1)
    long1 = math.radians(long1)
    lat2 = math.radians(lat2)
    long2 = math.radians(long2)
  
    # Calculate the bearing
    bearing = math.atan2(
        math.sin(long2 - long1) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1)
    )
  
    # Convert the bearing to degrees
    bearing = math.degrees(bearing)
    
    # Make sure the bearing is positive
    bearing = (bearing + 360) % 360
    
    return bearing


class CrosssectionsGenerator:
    """ A class for generating cross-sections
    """

    def __init__(self, config):
        """ Initialise an instance of CrosssectionsGenerator

            Parameters
            ----------
                config: dict
                    Configuration
        """

        # Retrieve loggger
        self._logger = logging.getLogger("eo-sw1d")

        # Store configuration
        self.config = config

        # Set internal GeoDataFrame to None
        self._gdf = None


    def process(self, resume: bool=False):
        """Run the main process

            Parameters
                resume: bool
                    True to resume from a previous run
        """

        dxs = self.config["dxs"]

        # Load centerlines shapefile
        centerlines_shp_file = self.config["centerlines-shapefile"]
        centerlines_gdf = gpd.read_file(centerlines_shp_file)

        # Load sections centers shapefile
        sections_nodes_shp_file = self.config["sections-nodes-shapefile"]
        sections_nodes_gdf = gpd.read_file(sections_nodes_shp_file)

        gdf_data = {"node_id": [], "reach_id": [], "nx": [], "ny": [], "width": []}
        gdf_geom = []
        for i in tqdm(range(0, centerlines_gdf.index.size)):

            centerlines_index = centerlines_gdf.index[i]
            reach_id = centerlines_gdf.loc[centerlines_index, "reach_id"]
            reach_length = centerlines_gdf.loc[centerlines_index, "reach_len"]
            reach_geom = centerlines_gdf.loc[centerlines_index, "geometry"]

            # Select nodes
            centerline_nodes = sections_nodes_gdf[sections_nodes_gdf["reach_id"] == reach_id]

            for j in range(0, centerline_nodes.index.size):

                node_index = centerline_nodes.index[j]
                node_geom = centerline_nodes.loc[node_index, "geometry"]
                node_width = centerline_nodes.loc[node_index, "width"]
                node_id = centerline_nodes.loc[node_index, "node_id"]
                alpha = reach_geom.project(node_geom, normalized=True)
                xsc = reach_length * alpha
                alpham = (xsc - dxs) / reach_length
                alphap = (xsc + dxs) / reach_length
                pointm = reach_geom.interpolate(alpham, normalized=True)
                pointp = reach_geom.interpolate(alphap, normalized=True)
                dx = pointp.x - pointm.x
                dy = pointp.y - pointm.y
                nx = -dy / np.sqrt(dx**2 + dy**2)
                ny = dx / np.sqrt(dx**2 + dy**2)
                along_bearing = calc_bearing(pointm.y, pointm.x, pointp.y, pointp.x)
                normal_bearing = (along_bearing + 90.0) % 360.0
                demi_width = 0.5 * node_width
                left = distance(meters=demi_width).destination((node_geom.y, node_geom.x), bearing=normal_bearing)
                right = distance(meters=demi_width).destination((node_geom.y, node_geom.x), bearing=(normal_bearing + 180.0) % 360.0)

                gdf_data["node_id"].append(node_id)
                gdf_data["reach_id"].append(reach_id)
                gdf_data["nx"].append(nx)
                gdf_data["ny"].append(ny)
                gdf_data["width"].append(node_width)

                gdf_geom.append(LineString([(right.longitude, right.latitude), (left.longitude, left.latitude)]))

        # Create the internal GeoDataFrame
        self._gdf = gpd.GeoDataFrame(data=gdf_data, geometry=gdf_geom, crs=4326)
        if "shapefile" in self.config["destination"]:
            self._gdf.to_file(self.config["destination"]["shapefile"])




