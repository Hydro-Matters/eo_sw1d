import geopandas as gpd
import glob
import logging
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as sps
from scipy import signal
from tqdm.autonotebook import tqdm
import warnings

try:
    from rt_river_segmentation.denoising import hydraulic_filtering
except:
    warnings.warn("Unable to import rt_river_segmentation. Hydraulic filtering is disabled")
    hydraulic_filtering = None


def lowpass_filter(x, y, filtername="butter", dx=10, xflow=5):
  
    # Interpolate values on regular grid if needed
    x = np.array(x)
    y = np.array(y)
    xreg = np.arange(x[0], x[-1], dx)
    yreg = np.interp(xreg, x, y)
    N = xreg.size
    
    xreg = np.concatenate((-xreg[::-1], xreg))
    yreg = np.concatenate((yreg[::-1], yreg))
    xreg = np.concatenate((xreg, xreg[-1]+xreg))
    yreg = np.concatenate((yreg, yreg[::-1]))
    
    # Compute period and number of samples
    T = xreg[-1] - xreg[0]
    fhigh = 1.0 / dx
    cutoff =  fhigh / xflow

    if filtername == "butter":
    
        nyq = 0.5 * fhigh
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(5, normal_cutoff, btype="low", analog=False)
        yfiltreg = signal.filtfilt(b, a, yreg, padtype="even")
        xreg = xreg[N:N*2]
        yfiltreg = yfiltreg[N:N*2]
        yfilt = np.interp(x, xreg, yfiltreg)
        return yfilt
  
    elif filtername == "bessel":
    
        nyq = 0.5 * fhigh
        normal_cutoff = cutoff / nyq
        b, a = signal.bessel(5, normal_cutoff, btype="low", analog=False)
        yfiltreg = signal.filtfilt(b, a, yreg, padtype="even")
        xreg = xreg[N:N*2]
        yfiltreg = yfiltreg[N:N*2]
        yfilt = np.interp(x, xreg, yfiltreg)
        return yfilt
  
    else:
    
        raise ValueError("filtername must be 'butter' or 'bessel'")


class MeshProcessor:
    """ A class for computing generic mesh data (sections and segments)
    """

    def __init__(self, config: dict):
        """ Initialise an instance of MeshProcessor

            Parameters
                config: dict
                    Configuration
        """
        
        self._logger = logging.getLogger("eo-sw1d")
        self.config = config

    def process(self, resume=False):

        self._logger.info("[START] MeshProcessor::process") 

        mesh_file_prefix = self.config["output"]["mesh-file-prefix"]
        cs_csv_file = "%s_sections.csv" % mesh_file_prefix
        seg_csv_file = "%s_segments.csv" % mesh_file_prefix
        if resume is True and os.path.isfile(cs_csv_file) and os.path.isfile(seg_csv_file):

            # TODO implement loading previously computed mesh file
            raise NotImplementedError("Resume previsously computed mesh file is not implemented")
            return

        # Load network file
        self._logger.info("Load network file")
        self._logger.info("- file: %s" % self.config["sources"]["network-file"])
        network_gdf = gpd.read_file(self.config["sources"]["network-file"])

        # Load nodes file
        self._logger.info("Load nodes file")
        self._logger.info("- file: %s" % self.config["sources"]["nodes-file"])
        nodes_gdf = gpd.read_file(self.config["sources"]["nodes-file"])
        nodes_gdf = nodes_gdf.rename(columns={"wse": "p_wse", "width": "p_width"})
        nodes_gdf["wsefilt"] = np.ones(nodes_gdf.index.size) * -9999.0
        nodes_gdf["wsefilt2"] = np.ones(nodes_gdf.index.size) * -9999.0
        # nodes_gdf["wse_90"] = np.ones(nodes_gdf.index.size) * -9999.0
        nodes_gdf["widthfilt"] = np.ones(nodes_gdf.index.size) * -9999.0
        nodes_gdf["widthfilt2"] = np.ones(nodes_gdf.index.size) * -9999.0
        # nodes_gdf["width_90"] = np.ones(nodes_gdf.index.size) * -9999.0
        nodes_gdf["bathy"] = np.ones(nodes_gdf.index.size) * -9999.0
        recompute_nodes_dist_out = False
        if "recompute-nodes-distances" in self.config["sources"]:
            recompute_nodes_dist_out = self.config["sources"]["recompute-nodes-distances"]
        # if recompute_nodes_dist_out:
        #     nodes_gdf = self._recompute_nodes_dist_out_(nodes_gdf, network_gdf)

        # Load profiles file
        self._logger.info("Load profiles file")
        self._logger.info("- file: %s" % self.config["sources"]["profiles-file"])
        profiles_gdf = gpd.read_file(self.config["sources"]["profiles-file"])

        # Load inflow files
        inflow_Q10 = {}
        inflow_Q50 = {}
        for discharge_def in self.config["inflow"]:
            if "hydro3-file" in discharge_def:
                discharge_data = pd.read_csv(discharge_def["hydro3-file"], sep=",", parse_dates=["Date (TU)"])
                # print(discharge_data)
                # choice = input()
                t = discharge_data["Date (TU)"].values
                q = discharge_data["Valeur (en m³/s)"].values
                Q10 = np.percentile(q, 10)
                Q50 = np.percentile(q, 50)
                inflow_Q10[discharge_def["segment-index"]] = Q10
                inflow_Q50[discharge_def["segment-index"]] = Q50
            else:
                inflow_Q10[discharge_def["segment-index"]] = discharge_def["qmean"]
                inflow_Q50[discharge_def["segment-index"]] = discharge_def["qmean"]

        # Loop on segments
        self._logger.info("Process segments")
        segs_idx = sorted(list(network_gdf["seg_idx"].unique()))
        # print(segs_idx)

        cs_list = []
        segs_dict = {}

        for iseg, seg_idx in enumerate(segs_idx):
            segs_dict[seg_idx] = {
                "index": iseg,
                "first_cs": -1,
                "ncs": -1,
                "us_segs": [],
                "ds_segs": []
            }

        for iseg, seg_idx in enumerate(segs_idx):
            segs_dict[seg_idx]["first_cs"] = len(cs_list)
            reaches_selected = network_gdf[network_gdf["seg_idx"] == seg_idx]
            reaches_selected = reaches_selected.sort_values(by="dist_out")
            self._logger.info("- Process segment: %i [%i-%i]" % (seg_idx, reaches_selected["reach_id"].values[-1], reaches_selected["reach_id"].values[0]))

            Q10 = inflow_Q10[seg_idx]
            Q50 = inflow_Q50[seg_idx]
            self._logger.info("  - Q10: %.3f m3/s" % Q10)
            self._logger.info("  - Q50: %.3f m3/s" % Q50)
            # print("reaches_selected=")
            # print(reaches_selected.loc[:, ["reach_id", "rch_id_dn", "dist_out"]])
            # choice = input()
            reaches_list = list(reaches_selected["reach_id"].values)

            # Find upstream reaches
            reaches_list_extended = reaches_list.copy()
            if reaches_selected.loc[reaches_selected.index[0], "n_rch_dn"] > 0:

                next_reach_id = int(reaches_selected.loc[reaches_selected.index[0], "rch_id_dn"].split(" ")[0])
                reaches_list_extended.insert(0, next_reach_id)

            if reaches_selected.loc[reaches_selected.index[-1], "n_rch_up"] > 0:

                prev_reach_id = int(reaches_selected.loc[reaches_selected.index[-1], "rch_id_up"].split(" ")[0])
                reaches_list_extended.append(prev_reach_id)

            # print(reaches_list)
            # print(reaches_list_extended)


            nodes_selected_filtering = nodes_gdf[nodes_gdf["reach_id"].isin(reaches_list_extended)]
            nodes_selected = nodes_gdf[nodes_gdf["reach_id"].isin(reaches_list)]

            if recompute_nodes_dist_out:
                debug_recompute = False
                # if seg_idx == 4:
                #     debug_recompute = True
                nodes_selected = self._recompute_nodes_dist_out_(nodes_selected, reaches_selected, debug=debug_recompute)
                nodes_gdf.loc[nodes_selected.index, "dist_out"] = nodes_selected.loc[nodes_selected.index, "dist_out"]
            nodes_selected = nodes_selected.sort_values(by="dist_out", ascending=True)
            # print("nodes_selected=", nodes_selected)
            # choice = input()

            nodes_selected_merge = pd.merge(nodes_selected_filtering, profiles_gdf, how="left", left_on="node_id", right_on="key", suffixes=["_left", "_right"])
            nodes_selected_merge = nodes_selected_merge.sort_values(by="dist_out", ascending=True)
            nodes_selected_merge["left_index"] = nodes_selected_filtering.index
            nodes_selected_merge.set_index("left_index", inplace=True)
            # print("nodes_selected_merge=")
            # print(nodes_selected_merge)
            # print(nodes_selected)
            # print(nodes_selected_merge.columns)
            # print(nodes_selected_merge.index)
            # # print(nodes_selected_merge.loc[:, ["node_id", "p_wse", "wse_10", "wse_90", "p_width", "width_10", "width_90"]])
            # # choice = input()

            xr_filt = nodes_selected_merge["dist_out"].values
            xr = nodes_selected["dist_out"].values
            dxi = 50.0
            N = int(np.floor((xr_filt[-1] - xr_filt[0]) / dxi))
            # print("reaches_list=", reaches_list)
            # print("N=", N)
            # print(nodes_selected_merge.loc[:, ["node_id", "dist_out"]])
            xi = np.linspace(xr_filt[0], xr_filt[0]+N*dxi, N+1, endpoint=True)


            wse_10 = nodes_selected_merge["wse_10"].values
            wse_50 = nodes_selected_merge["wse_50"].values
            wse_90 = nodes_selected_merge["wse_90"].values
            # wsei = np.interp(xi, xr[np.isfinite(wse_10)], wse)
            # wseifilt = hydraulic_filtering(xi, wsei, 3)
            # wsefilt = np.interp(xr, xi, wseifilt)
            
            p_wse = nodes_selected_merge["p_wse"].values
            width_10 = nodes_selected_merge["width_10"].values
            width_50 = nodes_selected_merge["width_50"].values
            width_90 = nodes_selected_merge["width_90"].values
            # width_90[width_90 < -999.0] = np.nan
            p_width = nodes_selected_merge["p_width"].values
            # wse = wse_10.copy()
            wse = wse_50
            # wse[np.isnan(wse)] = p_wse[np.isnan(wse)]
            width = width_50
            width[np.isnan(width)] = p_width[np.isnan(width)]
            width[width<-999] = p_width[width<-999]

            if np.sum(np.isfinite(wse)) < 5:
                wse = p_wse
            # print("xi=", xi)
            wsei = np.interp(xi, xr_filt[np.isfinite(wse)], wse[np.isfinite(wse)])
            plot_steps = False
            # if seg_idx == 7:
            #     plot_steps = True
            if hydraulic_filtering is not None:
                wseifilt = hydraulic_filtering(xi, wsei, 3, plot_steps=plot_steps)
            else:
                wseifilt = wsei
            if "plot-dir" in self.config["output"]:
                fig = plt.figure()
                plt.plot(xi * 0.001, wsei)
                # plt.plot(xr_filt * 0.001, wse_10, "g--")
                plt.plot(xr_filt * 0.001, wse, "b.", label="source")
                # plt.plot(xr_filt * 0.001, wse_90, "c--")
                # plt.plot(xr_filt * 0.001, p_wse, "k--")
                plt.axvline(xr[0] * 0.001, ls="--", color="k")
                plt.axvline(xr[-1] * 0.001, ls="--", color="k")
                plt.plot(xi * 0.001, wseifilt, "r--", label="filtered")
                plt.legend()
                plt.xlabel("x (km)")
                plt.ylabel("wse (m)")
                plt.savefig(os.path.join(self.config["output"]["plot-dir"], "mesh_filt_seg%04i.png" % seg_idx))
                plt.close(fig)

            # xr = nodes_selected["dist_out"].values

            wsefilt = np.interp(xr, xi, wseifilt)

            nfilt = int(250/dxi)
            if nfilt%2 == 0:
                nfilt += 1
            widthi = np.interp(xi, xr_filt[np.isfinite(width)], width[np.isfinite(width)])
            # widthi_filtered = np.convolve(widthi, np.ones(nfilt) * (1.0/nfilt), mode='same')
            widthi_filtered = lowpass_filter(xi, widthi, dx=dxi, xflow=200)
            width_filtered = np.interp(xr, xi, widthi_filtered)
            width_filtered[width_filtered < 10.0] = 10.0
            # plt.plot(xr, width)
            # plt.plot(xi, widthi_filtered)
            # plt.show()


            # print(wsefilt)

            # Invert Manning equation
            # Q = K * (w * (H-b))**5/3 * W**(-2/3) * S**0.5
            # => (Q/K * W**(2/3) / (S**0.5))**(3/5) = w * (H-b)
            # => w b = w * H - (Q/K * W**(2/3) / (S**0.5))**(3/5)
            # => b = H - (Q/K * W**(2/3) / (S**0.5))**(3/5) / w
            reg = sps.linregress(xr, wsefilt)
            # h = (Q10 / 30.0 * width**(2./3.) / np.sqrt(reg.slope))**(3./5.) / width
            h = (Q10 / 30.0 * width_filtered**(2./3.) / np.sqrt(reg.slope))**(3./5.) / width_filtered
            bathy = wsefilt - h
            self._logger.info("  - h: [%.3f - %.3f] m" % (np.min(h), np.max(h)))

            bathyi = np.interp(xi, xr, bathy)
            # widthi_filtered = np.convolve(widthi, np.ones(nfilt) * (1.0/nfilt), mode='same')
            bathyi_filtered = lowpass_filter(xi, bathyi, dx=dxi, xflow=100)
            bathy_filtered = np.interp(xr, xi, bathyi_filtered)


            # nodes_selected = nodes_selected_base
            nodes_selected = nodes_selected.sort_values(by="dist_out")

            nodes_gdf.loc[nodes_selected.index, "wsefilt"] = wsefilt
            # nodes_gdf.loc[nodes_selected.index, "wsefilt2"] = wse_90
            # nodes_gdf.loc[nodes_selected.index, "wse_90"] = wse_90
            nodes_gdf.loc[nodes_selected.index, "widthfilt"] = width_filtered
            # nodes_gdf.loc[nodes_selected.index, "widthfilt2"] = width_90
            # nodes_gdf.loc[nodes_selected.index, "width_90"] = width_90
            nodes_gdf.loc[nodes_selected.index, "bathy"] = bathy
            nodes_gdf.loc[nodes_selected.index, "bathy_filtered"] = bathy_filtered

            # plt.plot(xr * 0.001, bathy, "k-", label="bathy")
            # plt.plot(xr * 0.001, wse_10, "b-", label="wse_10")
            # plt.plot(nodes_gdf.loc[nodes_selected.index, "dist_out"] * 0.001, nodes_gdf.loc[nodes_selected.index, "bathy"], "g--", label="bathy")
            # plt.plot(nodes_gdf.loc[nodes_selected.index, "dist_out"] * 0.001, nodes_gdf.loc[nodes_selected.index, "wsefilt"], "r--", label="wse10")
            # plt.show()


            # print(nodes_gdf.loc[nodes_selected.index, ["dist_out", "bathy", "wsefilt", "widthfilt"]])
            # choice = input()

            # if seg_idx >= 9:

            #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,9))
            #     ax1.plot(xr[::-1], "k-")
            #     ax1.plot(nodes_gdf.loc[nodes_selected.index[::-1], "dist_out"].values, "k+")
            #     ax2.plot(bathy[::-1], "k-")
            #     ax2.plot(nodes_gdf.loc[nodes_selected.index[::-1], "bathy"].values, "k+")
            #     ax2.plot(wsefilt[::-1], "b-")
            #     ax2.plot(nodes_gdf.loc[nodes_selected.index[::-1], "wsefilt"].values, "b+")
            #     ax2.plot(nodes_gdf.loc[nodes_selected.index[::-1], "wsefilt2"].values, "g--")
            #     # ax2.plot(nodes_selected_merge["wse_90"].values, "g--")
            #     ax3.plot(width[::-1], "b-")
            #     ax3.plot(nodes_gdf.loc[nodes_selected.index[::-1], "widthfilt"].values, "b+")
            #     # width_90 = nodes_selected_merge.loc[nodes_selected.index[::-1], "width_90"].values
            #     width_90 = nodes_gdf.loc[nodes_selected.index[::-1], "widthfilt2"].values
            #     width_90[width_90 < -999.0] = np.nan
            #     ax3.plot(width_90, "g--")
            #     plt.show()


            for index in nodes_selected.index[::-1]:
                # wse_90_cs = nodes_selected_merge.loc[index, "wse_90"]
                wse_90_cs = nodes_gdf.loc[index, "wsefilt2"]
                if wse_90_cs < nodes_gdf.loc[index, "wsefilt"]:
                #     wse_90_cs = wse_90_cs + nodes_gdf.loc[index, "wsefilt"] - nodes_selected_merge.loc[index, "wse_10"]
                    wse_90_cs = np.nan
                wse_90_cs = np.nan
                # width_90_cs = nodes_selected_merge.loc[index, "width_90"]
                width_90_cs = nodes_gdf.loc[index, "widthfilt2"]
                if np.isfinite(wse_90_cs) and np.isfinite(width_90_cs) and width_90_cs > -999.0 and width_90_cs > nodes_gdf.loc[index, "widthfilt"]:
                    cs_list.append({
                        "dist": nodes_gdf.loc[index, "dist_out"],
                        "x": nodes_gdf.loc[index, "geometry"].x,
                        "y": nodes_gdf.loc[index, "geometry"].y,
                        "bathy": nodes_gdf.loc[index, "bathy_filtered"],
                        # "wse": [nodes_gdf.loc[index, "wsefilt"], wse_90_cs],
                        # "width": [nodes_gdf.loc[index, "widthfilt"], width_90_cs]
                        "wse": [0.5 * (nodes_gdf.loc[index, "wsefilt"] + wse_90_cs)],
                        "width": [0.5 * (nodes_gdf.loc[index, "widthfilt"] + width_90_cs)],
                        "node_id": nodes_gdf.loc[index, "node_id"],
                        "seg_idx": seg_idx,
                        "seg_index": iseg
                    })
                else:
                    cs_list.append({
                        "dist": nodes_gdf.loc[index, "dist_out"],
                        "x": nodes_gdf.loc[index, "geometry"].x,
                        "y": nodes_gdf.loc[index, "geometry"].y,
                        "bathy": nodes_gdf.loc[index, "bathy_filtered"],
                        "wse": [nodes_gdf.loc[index, "wsefilt"]],
                        "width": [nodes_gdf.loc[index, "widthfilt"]],
                        "node_id": nodes_gdf.loc[index, "node_id"],
                        "seg_idx": seg_idx,
                        "seg_index": iseg
                    })

            segs_dict[seg_idx]["ncs"] = len(cs_list) - segs_dict[seg_idx]["first_cs"]

            if "plot-dir" in self.config["output"]:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
                ax1.plot(xr_filt * 0.001, wse_10, "b-+", label="wse Q10 (raw)")
                ax1.plot(xr_filt * 0.001, wse, "g--", label="wse Q10 (filled with p_wse)")
                ax1.plot(xr * 0.001, wsefilt, "r--", label="wse Q10 (filtered)")
                ax1.plot(xr * 0.001, bathy, "k-", label="bathy")
                ax1.plot(xr * 0.001, bathy_filtered, color="violet", ls="--", label="bathy (filtered)")
                ax1.axvline(xr[0] * 0.001, ls="--", color="k")
                ax1.axvline(xr[-1] * 0.001, ls="--", color="k")
                ax1.set_ylabel("wse (m)")
                ax1.legend()
                width_10_corrected = width_10.copy()
                width_10_corrected[width_10_corrected < -99] = np.nan
                ax2.plot(xr_filt * 0.001, width_10_corrected, label="with Q10 (raw)")
                ax2.plot(xr_filt * 0.001, width, label="width Q10 (filled with p_width)")
                ax2.set_ylabel("width (m)")
                ax3.plot(xr * 0.001, h, "r-")
                ax2.axvline(xr[0] * 0.001, ls="--", color="k")
                ax2.axvline(xr[-1] * 0.001, ls="--", color="k")
                ax3.set_ylabel("h (m)")
                ax3.set_xlabel("dist_out (km)")
                plt.savefig(os.path.join(self.config["output"]["plot-dir"], "mesh_profile_seg%04i.png" % seg_idx))
                plt.close(fig)

            # Update downstream segment Q10 and connectivities
            # print(reaches_selected.loc[:, ["reach_id", "dist_out", "n_rch_dn"]])
            # print("seg_idx=", seg_idx, "n_rch_dn=", reaches_selected.loc[reaches_selected.index[0], "n_rch_dn"])
            if reaches_selected.loc[reaches_selected.index[0], "n_rch_dn"] > 0:

                next_reach_ids = map(int, reaches_selected.loc[reaches_selected.index[0], "rch_id_dn"].split(" "))

                for next_reach_id in next_reach_ids:

                    # next_reach_id = int(reaches_selected.loc[reaches_selected.index[0], "rch_id_dn"])
                    next_reach_row = network_gdf[network_gdf["reach_id"] == next_reach_id]
                    next_segment_index = next_reach_row.loc[next_reach_row.index[0], "seg_idx"]
                    # print("seg_idx=", seg_idx, "next_reach_id=", next_reach_id, "segment_index=", next_segment_index)
                    if next_segment_index in inflow_Q10:
                        inflow_Q10[next_segment_index] += inflow_Q10[seg_idx]
                        inflow_Q50[next_segment_index] += inflow_Q50[seg_idx]
                    else:
                        inflow_Q10[next_segment_index] = inflow_Q10[seg_idx]
                        inflow_Q50[next_segment_index] = inflow_Q50[seg_idx]

                    segs_dict[next_segment_index]["us_segs"].append(segs_dict[seg_idx]["index"])
                    segs_dict[seg_idx]["ds_segs"].append(segs_dict[next_segment_index]["index"])

        # Create mesh shapefile
        if "mesh-shapefile" in self.config["output"]:
            mesh_shp_file = self.config["output"]["mesh-shapefile"]
        else:
            mesh_shp_file = "%s.shp" % os.path.splitext(self.config["output"]["mesh-file"])[0]
        mesh_shp_data = {"id": [], "xs": [], "x": [], "y": [], "bathy": [], "seg_id": [], "seg_index": []}
        for ics in range(0, len(cs_list)):
            mesh_shp_data["id"].append(cs_list[ics]["node_id"])
            mesh_shp_data["xs"].append(cs_list[ics]["dist"])
            mesh_shp_data["x"].append(cs_list[ics]["x"])
            mesh_shp_data["y"].append(cs_list[ics]["y"])
            mesh_shp_data["bathy"].append(cs_list[ics]["bathy"])
            mesh_shp_data["seg_id"].append(cs_list[ics]["seg_idx"])
            mesh_shp_data["seg_index"].append(cs_list[ics]["seg_index"])
        mesh_shp_gdf = gpd.GeoDataFrame(data=mesh_shp_data, geometry=gpd.points_from_xy(mesh_shp_data["x"], mesh_shp_data["y"]), crs=4326)
        mesh_shp_gdf.to_file(mesh_shp_file)

        # Create mesh csv
        cs_df_data = {"id": [], "seg_index": [], "xs": [], "x": [], "y": [], "bathy": [], "wse": [], "width": []}
        for ics in range(0, len(cs_list)):
            count = len(cs_list[ics]["wse"])
            cs_df_data["id"] += [cs_list[ics]["node_id"]] * count
            cs_df_data["seg_index"] += [cs_list[ics]["seg_index"]] * count
            cs_df_data["xs"] += [cs_list[ics]["dist"]] * count
            cs_df_data["x"] += [cs_list[ics]["x"]] * count
            cs_df_data["y"] += [cs_list[ics]["y"]] * count
            cs_df_data["bathy"] += [cs_list[ics]["bathy"]] * count
            cs_df_data["wse"] += cs_list[ics]["wse"]
            cs_df_data["width"] += cs_list[ics]["width"]
        self._cs_df = pd.DataFrame(data=cs_df_data)
        self._cs_df.to_csv(cs_csv_file, index=False, sep=";")

        seg_df_data = {"id": [], "first_cs": [], "last_cs": [], "cs_count": [], "us_segs": [], "ds_segs": []}
        for iseg, seg_idx in enumerate(segs_idx):
            seg_dict = segs_dict[seg_idx]
            seg_df_data["id"].append(seg_idx)
            seg_df_data["first_cs"].append(seg_dict["first_cs"])
            seg_df_data["last_cs"].append(seg_dict["first_cs"] + seg_dict["ncs"] - 1)
            seg_df_data["cs_count"].append(seg_dict["ncs"])
            if len(seg_dict["us_segs"]) == 0:
                seg_df_data["us_segs"].append("-1")
            else:
                seg_df_data["us_segs"].append(",".join(["%i" % us_seg for us_seg in seg_dict["us_segs"]]))
            if len(seg_dict["ds_segs"]) == 0:
                seg_df_data["ds_segs"].append("-1")
            else:
                seg_df_data["ds_segs"].append(",".join(["%i" % ds_seg for ds_seg in seg_dict["ds_segs"]]))

        self._seg_df = pd.DataFrame(data=seg_df_data)
        self._seg_df.to_csv(seg_csv_file, index=False, sep=";")

        # TODO Bathy correction at junctions

        self._logger.info("[ END ] MeshProcessor::process") 

    def _recompute_nodes_dist_out_(self, nodes_gdf, network_gdf, debug=False):

        # nodes_gdf["dist_out_orig"] = nodes_gdf["dist_out"].copy()
        nodes_gdf.assign(dist_out_orig=nodes_gdf.dist_out)
        reach_id_prev = None
        indices_list = []
        for index in nodes_gdf.index:
            node_id = nodes_gdf.loc[index, "node_id"]
            reach_id = nodes_gdf.loc[index, "reach_id"]
            if reach_id != reach_id_prev:
                if len(indices_list) > 0 and debug:
                    plt.plot(nodes_gdf.loc[indices_list, "dist_out_orig"], "r-")
                    plt.plot(nodes_gdf.loc[indices_list, "dist_out"], "b-")
                    plt.title("zoom")
                    plt.show()
                if reach_id_prev is not None and debug:
                    print("- node_dist: [%f, %f]" % (node_dist_min * 0.001, node_dist_max * 0.001))
                reach_id_prev = reach_id
                if debug:
                    print("index:%i, reach_id:%i, node_id:%i" % (index, reach_id, node_id))
                node_dist_min = np.inf
                node_dist_max = -np.inf
                indices_list = []
            indices_list.append(index)
            node_geometry = nodes_gdf.loc[index, "geometry"]
            reach_rows = network_gdf[network_gdf["reach_id"] == reach_id]
            # print("node_id=%i, reach_id=%i, found: %i" % (node_id, reach_id, reach_rows.index.size))
            reach_geometry = network_gdf.loc[reach_rows.index[0], "geometry"]
            reach_dist_out = network_gdf.loc[reach_rows.index[0], "dist_out"]
            reach_length = network_gdf.loc[reach_rows.index[0], "reach_len"]
            alpha = reach_geometry.project(node_geometry, normalized=True)
            nodes_gdf.loc[index, "dist_out"] = reach_dist_out - (1.0 - alpha) * reach_length
            if debug:
                print("  - [%i], node_id=%i, alpha=%f, dist_out = %f => %f" % (index, node_id, alpha, nodes_gdf.loc[index, "dist_out_orig"], nodes_gdf.loc[index, "dist_out"]))
            node_dist_min = min(nodes_gdf.loc[index, "dist_out"], node_dist_min)
            node_dist_max = max(nodes_gdf.loc[index, "dist_out"], node_dist_max)
            # nodes_gdf.loc[index, "dist_out_orig"] = reach_dist_out - (1.0 - alpha) * reach_length
        
        nodes_gdf = nodes_gdf[np.isfinite(nodes_gdf["dist_out"])]
        if debug:
            print("- node_dist: [%f, %f]" % (node_dist_min * 0.001, node_dist_max * 0.001))

            plt.plot(nodes_gdf["dist_out_orig"].values)
            plt.plot(nodes_gdf["dist_out"].values)
            plt.show()



        return nodes_gdf
