# %%
import climnet.community_detection.cd_functions as cdf
import geoutils.utils.spatial_utils as sput
import xarray as xr
from matplotlib.patches import Rectangle
import geoutils.utils.general_utils as gut
import climnet.community_detection.membership_likelihood as ml
import os
import climnet.datasets.evs_dataset as eds
from importlib import reload
import numpy as np
import geoutils.plotting.plots as cplt
import climnet.network.clim_networkx as cn

name = "mswep"
grid_type = "fekete"
grid_step = 1

output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/plots/"

# %%
# Load Network file EE
reload(eds)
q_ee = .9
output_folder = "summer_monsoon"
name_prefix = f"{name}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"

reload(eds)
q_sig = 0.95
lat_range = [-30, 65]
lb = False
if lb:
    nx_path_file = output_dir + \
        f"{output_folder}/{name_prefix}_{q_sig}_lat_{lat_range}_lb_ES_nx.gml.gz"
    dataset_file = output_dir + \
        f"/{output_folder}/{name_prefix}_1979_2021_lat_{lat_range}_lb_ds.nc"
else:
    nx_path_file = output_dir + \
        f"{output_folder}/{name_prefix}_{q_sig}_lat_{lat_range}_ES_nx.gml.gz"
    dataset_file = output_dir + \
        f"/{output_folder}/{name_prefix}_1979_2021_lat_{lat_range}_ds.nc"

ds = eds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False
)
# %%
reload(cn)
cnx = cn.Clim_NetworkX(dataset=ds, nx_path_file=nx_path_file)
# %%
reload(cdf)
cd = 'gt'
sp_arr = []

for run in np.arange(110):
    if cd == 'gt':
        cd_folder = 'graph_tool'
        B_max = 10
        if lb:
            sp_theta = (
                plot_dir
                + f"{output_folder}/graph_tool/lb/{run}_{name_prefix}_{q_sig}_{B_max}_lb.npy"
            )

        else:
            sp_theta = (
                plot_dir
                + f"{output_folder}/graph_tool/{run}_{name_prefix}_{q_sig}_{B_max}.npy"
            )
    elif cd == 'nk':
        cd_folder = 'nk'
        sp_theta = (
            plot_dir +
            f"{output_folder}/{cd_folder}/{run}_{name_prefix}_{q_sig}.npy"
        )
    if os.path.exists(sp_theta):
        sp_arr.append(sp_theta)

theta_arr = cdf.load_sp_arr(sp_arr=sp_arr)

# %%
cnx.ds.add_loc_dict(
    name="EIO",
    lname='Equatorial Indian Ocean',
    lon_range=(60, 75),
    lat_range=(0, 10),
    color="tab:red",
    n_rep_ids=3,
    reset_loc_dict=True
)

cnx.ds.add_loc_dict(
    name="WP",
    lname='West Pacific',
    lon_range=[145, 165],
    lat_range=[20, 30],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="NC",
    lname='North China',
    lon_range=[120, 135],
    lat_range=[32, 47],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="NI",
    lname='North India',
    lon_range=[76, 87],
    lat_range=[22, 32],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="BSISO",
    lname='BSISO',
    lon_range=(75, 80),
    lat_range=[15, 22],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="SZ",
    lname='Sahel Zone',
    lon_range=[-10, 22],
    lat_range=[10, 15],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="NAM",
    lname='North America',
    lon_range=(-90, -70),
    lat_range=[5, 15],
    color="tab:red",
    n_rep_ids=3,
)

# %%
reload(cplt)
loc_map = []
region_dict = cnx.ds.loc_dict
for mname, mregion in region_dict.items():
    # loc_map.append(mregion["locs"])
    loc_map.append(cnx.ds.get_locs_for_indices(mregion['rep_ids']))
loc_map = np.concatenate(loc_map, axis=0)

ax = cplt.plot_map(
    loc_map,
    z=np.ones(len(loc_map)),
    # central_longitude=180,
    projection="PlateCarree",
    plt_grid=True,
    plot_type="points",
    vmin=0,
    vmax=1,
    title="Selected Rep Ids",
    label='Points',
    alpha=1,
    lon_range=[-180, 180],
    lat_range=[-20, 60],
)
savepath = plot_dir + \
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_pids.png"

# cplt.save_fig(savepath)

# %%
reload(cplt)
for run in [0]:

    hard_cluster = theta_arr[run]['hard_cluster']
    hc_map = cnx.ds.get_map(hard_cluster)
    im = cplt.plot_map(
        hc_map,
        ds=cnx.ds,
        cmap='Paired',
        central_longitude=70,
        tick_step=1,
        levels=10,
        vmin=0, vmax=10,
        plot_type="contourf",
        significance_mask=True,
        projection="Robinson",
        extend="neither",
        label="Group number",
        orientation="horizontal",
        shift_ticks=True,
        set_int=True,
    )

    savepath = (
        plot_dir
        + f"{output_folder}/{cd_folder}/{run}_hard_clusters_{name_prefix}_{B_max}_{q_sig}.png"
    )
    cplt.save_fig(savepath=savepath)

# %%
reload(ml)
reload(gut)
reload(cplt)

res_dict = ml.get_prob_maps_community(
    ds=cnx.ds,
    theta_arr=theta_arr,
    sig_th=0.7,
    exclude_outlayers=True,
)
if lb:
    savepath = (
        plot_dir +
        f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_lb_prob_maps.npy"
    )
else:
    savepath = (
        plot_dir +
        f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps_global.npy"
    )

# fut.save_np_dict(res_dict, savepath)
# %%
# Single member likelihoods for one community:
reload(cplt)
region = 0
gr_map = res_dict[region]['prob_map']

im = cplt.plot_map(
    gr_map,
    ds=cnx.ds,
    significance_mask=True,
    central_longitude=70,
    plot_type="colormesh",
    cmap="Reds",
    levels=10,
    vmin=0,
    vmax=1,
    # title=f"Membership Likelihood {region}",
    projection="Robinson",
    extend="neither",
    label="Membership Likelihood",
    orientation="horizontal",
    round_dec=3,
)

this_map = res_dict[region]["map"]
color = 'k'
im = cplt.plot_map(
    this_map,
    ax=im['ax'],
    plot_type='contour',
    color=color,
    levels=0, vmin=0, vmax=1,
    lw=2,
    alpha=1,
    set_map=False)

savepath = (
    plot_dir
    + f"{output_folder}/{cd_folder}/msl_{region}_{q_sig}_{B_max}.png"
)
cplt.save_fig(savepath)


# %%
reload(cplt)

im = cplt.create_multi_plot(nrows=2, ncols=5,
                            projection='PlateCarree',
                            central_longitude=50,)
for idx, region in enumerate(np.arange(0, 10)):
    gr_map = res_dict[region]['prob_map']
    im_ml = cplt.plot_map(
        gr_map,
        ax=im['ax'][idx],
        ds=cnx.ds,
        significance_mask=True,
        plot_type="contourf",
        cmap="Reds",
        levels=10,
        vmin=0,
        vmax=1,
        title=f"Membership Likelihood {region}",
        extend="neither",
        label="Membership Likelihood",
        orientation="horizontal",
        round_dec=3,
    )

    this_map = res_dict[region]["map"]
    color = 'k'
    cplt.plot_map(
        this_map,
        ax=im_ml['ax'],
        plot_type='contour',
        color=color,
        levels=0, vmin=0, vmax=1,
        lw=1,
        alpha=1,
        set_map=False)

savepath = (
    plot_dir
    + f"{output_folder}/{cd_folder}/msl_all_{q_sig}_{B_max}.png"
)
cplt.save_fig(savepath)
# %%
# Plot Density of points
data = res_dict[region]['map']
deg, rad, idx_map = cnx.ds.get_coordinates_flatten()
# %%
reload(sput)
kde_map = sput.get_kde_map(ds=cnx.ds, data=data, coord_rad=rad,
                           bandwidth=None)

im = cplt.plot_map(
    kde_map,
    ds=cnx.ds,
    significance_mask=True,
    plot_type="contourf",
    cmap="Reds",
    levels=10,
    vmin=1.,
    vmax=2.5,
    title=f"Density {region}",
    projection="PlateCarree",
    bar=True,
    plt_grid=True,
    label="Membership Likelihood",
    orientation="horizontal",
    round_dec=3,
)
savepath = (
    plot_dir
    + f"{output_folder}/{cd_folder}/density_of_points_{region}_{q_sig}_{B_max}.png"
)
cplt.save_fig(savepath)

# %%
# Plot Msl of all communities
reload(cplt)
ax = None
legend_items = []
legend_item_names = []
regions = list(res_dict.keys())
for idx, region in enumerate(regions):

    this_map = res_dict[region]["map"]
    color = cplt.colors[idx]
    im = cplt.plot_map(
        this_map,
        ax=ax,
        projection='PlateCarree' if idx == 0 else None,
        figsize=(9, 7),
        plot_type='contour',
        color=color,
        levels=0, vmin=0, vmax=1,
        bar=False,
        lw=3,
        alpha=1,
        set_map=False if idx > 0 else True)

    im = cplt.plot_map(
        xr.where(this_map == 1, 1, np.nan),
        # this_map,
        ds=cnx.ds,
        significance_mask=True,
        ax=im['ax'],
        plt_grid=False,
        plot_type='contourf',
        color=color,
        cmap=None,
        levels=2,
        vmin=0, vmax=1,
        bar=False,
        alpha=0.6
    )
    ax = im['ax']
    legend_items.append(Rectangle((0, 0), 1, 1,
                                  fc=color, alpha=0.5,
                                  fill=True,
                                  edgecolor=color,
                                  linewidth=2))
    legend_item_names.append(f"{region}")

cplt.set_legend(ax=im['ax'],
                legend_items=legend_items,
                label_arr=legend_item_names,
                loc='outside',
                ncol_legend=3,
                box_loc=(0, 0))

savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name}_{cd}_msl_all_regions_{q_sig}.png"
)
cplt.save_fig(savepath=savepath)
# %%
reload(cplt)
nrows = 2
ncols = 3
im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.25,
                            wspace=0.25,
                            end_idx=len(regions))

for idx, region in enumerate(regions):

    # EE TS
    this_dict = res_dict[region]
    prob_map = this_dict['prob_map']

    im_comp = cplt.plot_map(
        prob_map,
        ds=cnx.ds,
        ax=im['ax'][idx],
        plot_type="contourf",
        title=f'{this_dict["lname"]}',
        ds_mask=True,
        cmap="Reds",
        levels=10,
        vmin=0,
        vmax=1,
        significance_mask=cnx.ds.mask,
        # title=f"Probability map for q={q_sig} {region}",
        y_title=1.1,
        extend="neither",
        bar=False,
        plt_grid=True,
    )


cbar = cplt.add_colorbar(im=im_comp,
                         fig=im['fig'],
                         x_pos=0.2,
                         y_pos=0.05, width=0.6, height=0.02,
                         orientation='horizontal',
                         label='Membership Likelihood',
                         tick_step=1,
                         )

savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name}_{cd}_msl_single_regions_{q_sig}.png"
)
cplt.save_fig(savepath,
              fig=im['fig'])

# %%
region = 'SZ'
loc_map = cnx.ds.get_locs_for_indices(res_dict[f'{region}']['ids'])

ax = cplt.plot_map(
    loc_map,
    z=np.ones(len(loc_map)),
    central_longitude=50,
    projection="PlateCarree",
    plt_grid=True,
    plot_type="points",
    vmin=0,
    vmax=1,
    title=f"Selected Ids {region}",
    label='Points',
    alpha=1,
    lat_range=[-20, 60],
    lon_range=[-180, 180]
)
savepath = plot_dir + \
    f"{output_folder}/{cd_folder}/{region}_{q_sig}_pids.png"
