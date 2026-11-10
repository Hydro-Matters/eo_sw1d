import geopandas as gpd
import glob
import logging
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.optimize as spo
from tqdm.autonotebook import tqdm
import zipfile

from .alti_merge_processor import AltiMergeProcessor
from .width_merge_processor import WidthMergeProcessor


def powerlaw_fit(H, W):
    """ Helper function that fit a powerlaw for the W(H) curve

        Parameters
            H: np.array
                Array of wse
            W: np.array
                Array of widths
    """

    def powerlawfunc(H, a, b, H0):
        """ Compute the results (width) of a powerlaw in H (wse)

            Parameters
                H: np.array
                    Array of wse
                a: float
                    Multiplicative parameter
                b: float
                    Exponent parameter
                H0: float
                    Lowest wse parameter

        """
        
        return a*(H-H0)**b

    # Fit powerlaw curve using scipy curve_fit
    popt, pcol = spo.curve_fit(powerlawfunc, H, W, p0=np.asarray([W[0]*0.5,5.1,H[0]-1.0]), maxfev=50000)

    return powerlawfunc(H, *popt)


class MultiSourcesProcessor:
    """ A class for merging multiple sources of EO data
    """

    def __init__(self, config: dict):
        """ Initialise an instance of MultiSourcesProcessor

            Parameters
                config: dict
                    Configuration
        """

        # Retrieve loggger
        self._logger = logging.getLogger("eo-sw1d")

        # Store configuration
        self.config = config

    def process(self, resume=(False, False)):
        """ Run the main process

            Parameters
                resume: bool
                    True to resume from a previous run
        """

        self._logger.info("[START] MultiSourcesProcessor::process") 

        # Retrieve common parameters
        if "min-obs-count" in self.config:
            min_obs_count = int(self.config["min-obs-count"])
        else:
            min_obs_count = 10

        # Merge altimetric (wse) sources
        self._logger.info("Load wse data")
        alti_config = self.config["alti-config"]
        alti_config["topology"] = self.config["topology"]
        self._alti_processor = AltiMergeProcessor(alti_config)
        self._alti_processor.process(resume=resume[0])

        # Merge width sources
        width_config = self.config["width-config"]
        if width_config is not None:
            self._logger.info("Load width data")
            width_config["topology"] = alti_config["topology"]
            self._width_processor = WidthMergeProcessor(width_config)
            self._width_processor.process(resume=resume[1])
        else:
            self._width_processor = None

        self._merge_and_summarize_(min_obs_count)

        self._logger.info("[END] MultiSourcesProcessor::process") 


    def _merge_and_summarize_(self, min_obs_count: int=10):
        """ Merge and summarize data

            Parameters
                min_obs_count: int
                    Minimum number of observations for a valid profile
        """

        self._logger.info("Merge wse and width data")

        self._merged_data = {
            "key": [],
            "x": [],
            "y": [],
            "wse": [],
            "width": []
        }

        self._final_summary = {
            "key": [],
            "x": [],
            "y": [],
            "wse_10": [],
            "wse_50": [],
            "wse_90": [],
            "width_10": [],
            "width_50": [],
            "width_90": [],
            "wse_obs": [],
            "width_obs": []
        }
        
        # Loop on cross-sections keys
        keys = list(set(self._alti_processor.merged_data["key"]))
        for key_index in tqdm(range(0, len(keys))):

            # Retrieve wse and width data
            key = keys[key_index]
            wse_data = self._alti_processor.get_data(key)
            if self._width_processor is None:
                width_data = wse_data
            else:
                width_data = self._width_processor.get_data(key)

            # Compute average location
            x = wse_data["x"].mean()
            y = wse_data["y"].mean()

            # Retrieve wse timeseries and priority
            t_wse = wse_data["t"].values
            wse_values = wse_data["wse"].values
            wse_priority = wse_data["priority"].values
            wse_sources = wse_data["source"].values
            t_wse = t_wse[np.isfinite(wse_values)]
            wse_priority = wse_priority[np.isfinite(wse_values)]
            wse_sources = wse_sources[np.isfinite(wse_values)]
            wse_values = wse_values[np.isfinite(wse_values)]

            # Restrict to minimum priority for wse
            priority_min = np.min(wse_priority)
            wse_values = wse_values[wse_priority == priority_min]
            wse_sources = wse_sources[wse_priority == priority_min]
            t_wse = t_wse[wse_priority == priority_min]

            # Compute percentiles of wse
            if wse_values.size < min_obs_count:
                wse_10 = np.nan
                wse_50 = np.nan
                wse_90 = np.nan
            else:
                wse_10 = np.percentile(wse_values, 10)
                wse_50 = np.percentile(wse_values, 50)
                wse_90 = np.percentile(wse_values, 90)
            wse_count = wse_values.size

            # Retrieve width timeseries and priority
            if "width" in wse_data:
                if width_data is not None:
                    t_width = np.concatenate((wse_data["t"].values, width_data["t"].values))
                    width_values = np.concatenate((wse_data["width"].values, width_data["width"].values))
                    width_sources = np.concatenate((wse_data["source"].values * (-1), width_data["source"].values))
                    width_priority = np.concatenate((wse_data["priority"].values, width_data["priority"].values))
                else:
                    t_width = wse_data["t"].values
                    width_values = wse_data["width"].values
                    width_sources = wse_data["source"].values * (-1)
                    width_priority = wse_data["priority"].values
            else:
                t_width = width_data["t"].values
                width_values = width_data["width"].values
                width_sources = width_data["source"].values
                width_priority = width_data["priority"].values
            t_width = t_width[np.isfinite(width_values)]
            width_priority = width_priority[np.isfinite(width_values)]
            width_sources = width_sources[np.isfinite(width_values)]
            width_values = width_values[np.isfinite(width_values)]

            # Restrict to minimum priority for width
            if width_priority.size > 0:
                priority_min = np.min(width_priority)
                width_values = width_values[width_priority == priority_min]
                width_sources = width_sources[width_priority == priority_min]
                t_width = t_width[width_priority == priority_min]

            # Compute percentiles of width
            if width_values.size < min_obs_count:
                width_10 = np.nan
                width_50 = np.nan
                width_90 = np.nan
            else:
                width_values = np.maximum(width_values, 30.0)
                width_10 = np.percentile(width_values, 10)
                width_50 = np.percentile(width_values, 10)
                width_90 = np.percentile(width_values, 90)
            width_count = width_values.size


            if wse_count > min_obs_count and width_count > min_obs_count:

                # Sort wse in time
                isort = np.argsort(t_wse)
                t_wse = t_wse[isort]
                wse = wse_values[isort]

                # Sort width in time
                isort = np.argsort(t_width)
                t_width = t_width[isort]
                width = width_values[isort]

                # Interpolate width on wse times and vice-versa
                dt_wse = (t_wse - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
                dt_width = (t_width - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
                wse2 = np.interp(dt_width, dt_wse, wse)
                wse_reg = wse2[np.argsort(wse2)]
                width_reg = width[np.argsort(wse2)]
                if "section-method" in self.config:
                    section_method = self.config["section-method"]
                else:
                    section_method = "percentiles"
                if section_method == "percentiles":
                    wse_percentiles = np.array([np.percentile(wse_reg, pct) for pct in range(10, 100, 10)])
                    width_percentiles = np.array([np.percentile(width_reg, pct) for pct in range(10, 100, 10)])
                    wse_reg = wse_percentiles
                    width_reg = width_percentiles
                elif section_method == "powerlaw":
                    width_reg = powerlaw_fit(wse_reg, width_reg)
                else:
                    raise ValueError("Wrong section method: %s" % self.config["section-method"])

                # Sections plots
                if "section-plots-prefix" in self.config["output"]:

                    # Make plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
                    ax1.plot(t_width, wse2, "b-")
                    ax1.plot(t_width, wse2, "b.")
                    ax1.set_xlabel("date")
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.set_ylabel("wse (m)")
                    ax1.yaxis.label.set_color('b')
                    ax1.tick_params(axis='y', colors='b')
                    twinx = ax1.twinx()
                    twinx.plot(t_width, width, "g-")
                    twinx.plot(t_width, width, "g.")
                    twinx.set_ylabel("W (m)")
                    twinx.yaxis.label.set_color('g')
                    twinx.tick_params(axis='y', colors='g')
                    ax2.set_xlabel("W (m)")
                    ax2.set_ylabel("wse (m)")

                    # Add percentiles to plot
                    wse_percentiles = np.array([np.percentile(wse, pct) for pct in range(10, 100, 10)])
                    wse2_percentiles = np.array([np.percentile(wse2, pct) for pct in range(10, 100, 10)])
                    width_percentiles = np.array([np.percentile(width, pct) for pct in range(10, 100, 10)])
                    # ax2.plot(-0.5 * width_percentiles, wse_percentiles, "b-")
                    # ax2.plot(0.5 * width_percentiles, wse_percentiles, "b-")
                    ax2.plot(-0.5 * width_percentiles, wse2_percentiles, "g-")
                    ax2.plot(0.5 * width_percentiles, wse2_percentiles, "g-")

                    # Add regression profiles
                    ax2.plot(-0.5 * width, wse2, "r.", alpha=0.5)
                    ax2.plot(0.5 * width, wse2, "r.", alpha=0.5)
                    ax2.plot(-0.5 * width_reg, wse_reg, "b--")
                    ax2.plot(0.5 * width_reg, wse_reg, "b--")

                    # Finalize and save plot
                    fig_title = "node_id: %i" % key
                    fig.suptitle(fig_title)
                    plt.tight_layout()
                    plot_file = "%s_%i.png" % (self.config["output"]["section-plots-prefix"], key)
                    fig.savefig(plot_file)
                    plt.close(fig)                

                # Update profiles
                for i in range(0, width_reg.size):
                    self._merged_data["key"].append(key)
                    self._merged_data["x"].append(x)
                    self._merged_data["y"].append(y)
                    self._merged_data["wse"].append(wse_reg[i])
                    self._merged_data["width"].append(width_reg[i])

            else:

                wse_reg = None
                self._merged_data["key"].append(key)
                self._merged_data["x"].append(x)
                self._merged_data["y"].append(y)
                self._merged_data["wse"].append(np.nan)
                self._merged_data["width"].append(np.nan)

            # Update summary lists
            self._final_summary["key"].append(key)
            self._final_summary["x"].append(x)
            self._final_summary["y"].append(y)
            self._final_summary["wse_10"].append(wse_10)
            self._final_summary["wse_50"].append(wse_50)
            self._final_summary["wse_90"].append(wse_90)
            self._final_summary["width_10"].append(width_10)
            self._final_summary["width_50"].append(width_50)
            self._final_summary["width_90"].append(width_90)
            self._final_summary["wse_obs"].append(wse_count)
            self._final_summary["width_obs"].append(width_count)

        # Create profiles GeoDataFrame and save to csv
        self._merged_gdf = pd.DataFrame(data=self._merged_data)
        if "data-file" in self.config["output"]:
            merged_csv_file = self.config["output"]["data-file"]
            self._merged_gdf.to_csv(merged_csv_file, sep=";", index=False)

        # Create summary GeoDataFrame and save to shapefile
        if "summary-shapefile" in self.config["output"]:
            summary_csv_file = self.config["output"]["summary-shapefile"]
            final_summary_gdf = gpd.GeoDataFrame(data=self._final_summary, 
                                                geometry=gpd.points_from_xy(self._final_summary["x"], self._final_summary["y"]),
                                                crs=self._alti_processor.crs)
            final_summary_gdf.to_file(summary_csv_file)

        self._logger.info("[ END ] MultiSourcesProcessor::process") 

        return self._final_summary
