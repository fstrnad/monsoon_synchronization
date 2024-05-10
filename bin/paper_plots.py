# %%
import geoutils.plotting.videos as vid
import geoutils.utils.met_utils as mut
import geoutils.geodata.multilevel_pressure as mp
import geoutils.geodata.wind_dataset as wds
import geoutils.indices.tej_index as tej
import geoutils.geodata.base_dataset as bds
import geoutils.indices.bsiso_index as bs
import geoutils.geodata.moist_static_energy as mse
import geoutils.tsa.time_series_analysis as tsa
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import xarray as xr
import geoutils.utils.general_utils as gut
import climnet.datasets.evs_dataset as eds
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import geoutils.plotting.plots as cplt
import climnet.network.clim_networkx as cn
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.indices.pdo_index as pdo
import geoutils.indices.mei_v2_index as mei

lat_range_cut = [-30, 70]
lon_range_cut = [-180, 180]


name = "mswep"
grid_type = "fekete"
grid_step = 1

output_dir = "/home/strnad/data/climnet/outputs/summer_monsoon/"
plot_dir = "/home/strnad/data/plots/summer_monsoon/"
data_dir = "/home/strnad/data/"
lat_range_cut = [-30, 70]
lon_range_cut = [-180, 180]

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
nx_path_file = output_dir + \
    f"{name_prefix}_{q_sig}_lat_{lat_range}_ES_nx.gml.gz"
dataset_file = output_dir + \
    f"{name_prefix}_1979_2021_lat_{lat_range}_ds.nc"

ds = eds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False
)

# %%
reload(cn)
cnx = cn.Clim_NetworkX(dataset=ds, nx_path_file=nx_path_file)
# %%
# Use time points via community output of graph_tool
B_max = 10
cd_folder = 'graph_tool'
q_sig = 0.95

savepath_loc = (
    plot_dir +
    f"/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps_global.npy"

)

comm_dict = fut.load_np_dict(savepath_loc)
# %%

reload(bds)
lev = 200
dataset_file_u = data_dir + \
    f"/climate_data/1/era5_u_{1}_{lev}_ds.nc"

u200 = bds.BaseDataset(data_nc=dataset_file_u,
                       can=True,
                       an_types=['dayofyear', 'month'],
                       lat_range=[0, 90],  # Only northern hemisphere
                       month_range=['Jun', 'Sep'],
                       init_mask=False)
# %%
reload(tej)
an_type = 'month'
var_type = f'an_{an_type}'
# var_type = 'u'
tej_index = tej.get_tej_index(u200.ds[var_type])
tej_tps = tej.get_tej_strength(u200=u200.ds[var_type],
                               definition='thresh',
                               tej_val=0)
tej_dict = dict(
    pos=tej_tps['enhanced'],
    neg=tej_tps['reduced'],
)
# %%
reload(tu)
reload(bs)

index_def = 'Kikuchi'
bsiso_index = bs.get_bsiso_index(
    # time_range=time_range,
    start_month='Jun',
    end_month='Sep',
    index_def=index_def)
var_bsiso1 = 'BSISO1'
var_bsiso2 = 'BSISO2'

bsiso1 = bsiso_index[var_bsiso1]
bsiso2 = bsiso_index[var_bsiso2]

# %%
# Compare synchronization of BSISO1 and BSISO2 and ENSO
reload(tu)
reload(gut)
reload(cplt)
reload(tsa)
reload(sut)
reload(bs)

savepath_meiv2 = '/home/strnad/data/meiv2/meiv2_days.nc'
mei_index = xr.open_dataset(savepath_meiv2)['MEIV2']
mei_index = tu.get_month_range_data(mei_index,
                                    start_month='Jul', end_month='Aug')
# All ENSO time points
nino_tps = mei_index[mei_index >= 0.5]
nina_tps = mei_index[mei_index <= -0.5]
neutral_tps = mei_index[(mei_index < 0.5) & (mei_index > -0.5)]

enso_types = {
    'El Nino': nino_tps,
    'La Nina': nina_tps,
    'Neutral': neutral_tps
}

savepath = plot_dir + \
    f"/trigger/separate_communities_ts_dict.npy"
ts_dict = fut.load_np_dict(savepath)
region = 'NISZ'

tps_inst_act = ts_dict[region]['active_years']
tps_inst_break = ts_dict[region]['break_years']
region = 'NISZ'

exclude_region = 'NISZ'
include_region = 'SZ'
comp_th = 0.95
# savepath = plot_dir + \
#     f"synchronization/synchronization_lag_exclude_{exclude_region}_{comp_th}_ts_dict.npy"

savepath = plot_dir + \
    f"synchronization/synchronization_lag_{10}_include_{include_region}_{comp_th}_ts_dict.npy"
lagged_sync_dict = fut.load_np_dict(savepath)
sregion = 'NI_SZ'
# sregion = 'SZ_NI'

lag = 10
# Synchronous and not synchronous days
tps_sync = lagged_sync_dict[sregion][lag]['tps_active']
tps_sync_ja = lagged_sync_dict[sregion][lag]['tps_sync_ja']
tps_sync_jjas = lagged_sync_dict[sregion][lag]['tps_sync_jjas']

break_years = lagged_sync_dict[sregion][lag]['break_years']
lagged_active_years = lagged_sync_dict[sregion][lag]['active_years']

background_tps = tu.get_periods_tps(tps=tps_sync,
                                    end=-10,
                                    start=-0)
tej_active_la_nina = tu.get_sel_tps_lst(ds=bsiso_index['BSISO-phase'],
                                        tps_lst=[nina_tps,
                                                 tej_dict['pos']], )
lagged_index = lagged_sync_dict[sregion][lag]['sync_ts']
tr_jjas = tu.get_dates_of_time_range(['1979-01-01',
                                      '2021-01-01'],
                                     start_month='Jun',
                                     end_month='Sep')

tr_ja = tu.get_dates_of_time_range(['1979-01-01',
                                    '2021-01-01'],
                                   start_month='Jul',
                                   end_month='Aug')
phase_num = 6
tps_dict = {
    'lagged_index': lagged_index,
    'tps_sync': tu.remove_consecutive_tps(tps_sync,
                                          #   start=10,
                                          #   steps=12,
                                          ),
    'tps_sync_ja': tu.remove_consecutive_tps(tps_sync_ja,
                                             #  start=10,
                                             #  steps=12
                                             ),
    'tps_sync_jjas': tu.remove_consecutive_tps(tps_sync_jjas,
                                               #    start=10,
                                               #    steps=12
                                               ),
    'background active': background_tps,
    'BSISO+ENSO+TEJ': tej_active_la_nina.where(tej_active_la_nina == phase_num, drop=True).time,
    'break years': tps_inst_break,
    'active years': tps_inst_act,
    'JJAS': tr_jjas,
    'JA': tr_ja,
}

# %% Figure 1

reload(cplt)
region = 'NI'
gr_map = comm_dict[region]['prob_map']
region1 = 'NI'
region2 = 'SZ'
region3 = 'EA'
regions = ['NI', 'SZ', 'EA']
colors = ['tab:red', 'tab:green', 'magenta']
# To get right contours
ts_dict['EA']['lon_range']=[90, 140]
ts_dict['EA']['lat_range']=[30, 65]

fig = plt.figure(
    figsize=(9, 9)
)

# set up ax1 and plot the communities and event series there
proj = cplt.get_projection(projection='PlateCarree',
                           )
# ax1 = fig.add_axes([0., 0.12, .5, 1], projection=proj)
# ax2 = fig.add_axes([0.7, 0.48, 0.7, 0.27])
# ax3 = fig.add_axes([0.0, 0.1, 0.55, 0.25])
# ax4 = fig.add_axes([0.7, 0.1, 0.75, 0.25])

ax1 = fig.add_axes([0., 0.12, .65, 1], projection=proj)
ax2 = fig.add_axes([0.8, 0.45, 0.65, 0.3])
ax3 = fig.add_axes([0.0, 0.05, 0.55, 0.25])
ax4 = fig.add_axes([0.7, 0.05, 0.75, 0.25])

axs = [ax1, ax2, ax3, ax4]
cplt.enumerate_subplots(axs=axs)

im = cplt.plot_map(
    xr.where(gr_map < 1, gr_map+0.05, gr_map),
    ax=ax1,
    ds=cnx.ds,
    significance_mask=True,
    plot_type="contourf",
    cmap="GnBu",
    levels=10,
    vmin=0,
    vmax=1,
    title=f"Membership likelihood community",
    y_title=1.15,
    plot_grid=True,
    extend="neither",
    label="Membership likelihood",
    orientation="horizontal",
    lon_range=[-50, 180],
    lat_range=[-20, 65],
)

for idx, this_region in enumerate([region1, region2, region3]):
    full_map = comm_dict[region]["map"]
    lon_range = np.array(ts_dict[this_region]['lon_range']) + [-2, 1]
    lat_range = np.array(ts_dict[this_region]['lat_range']) + [-2, 2]

    this_map = sput.get_locations_in_range(full_map,
                                           lon_range=lon_range,
                                           lat_range=lat_range)

    im = cplt.plot_map(
        this_map,
        ax=ax1,
        plot_type='contour',
        color=colors[idx],
        levels=1, vmin=0, vmax=1,
        lw=3,
        alpha=1,
        zorder=11,
        set_map=False)
    cplt.plt_text(ax=ax1,
                  text=this_region,
                  geoaxis=True,
                  color=colors[idx],
                  xpos=np.mean(lon_range),
                  ypos=np.min(lat_range)-7 if idx <2 else np.min(lat_range)-4,
                  zorder=12,
                  weight='bold',
                  fsize=16,
                  )

ts1 = ts_dict[region1]['ts']
ts2 = ts_dict[region2]['ts']
ts3 = ts_dict[region3]['ts']

timemean = 'day'
maxlags = 40
cutoff = 5
p = 0.05
for idx, ts_lag in enumerate([ts2, ts3]):
    ll_dict1 = tu.lead_lag_corr(ts1=ts1, ts2=ts_lag,
                                maxlags=maxlags,
                                cutoff=cutoff,
                                corr_method='spearman',
                                )
    tau_lag = ll_dict1['tau']
    lag_corr1 = ll_dict1['corr']
    p_vals = ll_dict1['p_val']
    im = cplt.plot_xy(
        ax=ax2,
        x_arr=[tau_lag
               ],
        y_arr=[lag_corr1,
               ],
        title=fr'Lead-lag correlation',
        y_title=1.1,
        xlabel=r'Time-Lag $\tau$'+f'[{timemean}s]',
        ylabel='Correlation',
        label_arr=[rf'NI $\rightarrow$ {regions[idx+1]}'],
        ylim=(-0.5, 0.55),
        lw_arr=[3],
        ls_arr=['-', '-', '-'],
        mk_arr=[None, None, None],
        stdize=False,
        set_grid=True,
        set_axis=True,
        color=colors[idx+1],
    )
    cplt.plot_hline(
        ax=ax2,
        y=p,
        color='black',
        ls='dashed',
        lw=2,)

    cplt.plot_hline(
        ax=ax2,
        y=-p,
        color='black',
        ls='dashed',
        lw=2,)

    cplt.plot_vline(
        ax=ax2,
        x=ll_dict1['tau_max'],
        color='black',
        ls='solid',
        lw=3,
    )
    cplt.plt_text(ax=ax2,
                  xpos=ll_dict1['tau_max']+0.5,
                  ypos=-0.02,
                  text=f'{timemean} {ll_dict1["tau_max"]}',
                  box=False
                  )


# Occurence of most sync days in each year
# Active and break years (all time points in JA)
tps_sync_jjas = tps_dict['tps_sync_jjas']
q_active_break = 0.75
yearly_tps = tu.count_time_points(time_points=tps_sync_jjas,
                                  freq='Y')
year_separte_arr = sut.get_values_above_val(dataarray=yearly_tps,
                                            q=q_active_break,)
sy = tu.get_sy_ey_time(times=yearly_tps.time)[0]
cplt.plot_xy(
    ax=ax3,
    x_arr=[yearly_tps.time,
           year_separte_arr['above'].time,
           year_separte_arr['below'].time,
           #    year_separte_arr['between'].time
           ],
    y_arr=[yearly_tps,
           year_separte_arr['above'],
           year_separte_arr['below'],
           #    year_separte_arr['between',
           ],
    label_arr=[None,
               None,  # 'Active Years',
               None,  # 'Break Years',
               'Between'],
    title=rf'Interannual variability',
    xlabel=f'Year (Start from {sy})',
    ylabel='No. of most sync. days',
    # ylim=[0, ],
    loc='outside',
    ls_arr=['-',
            'None',
            'None'],
    mk_arr=['None',
            'x', 'x'],
    stdize=False,
    ts_axis=False,
    set_legend=False,
    set_axis=True,
)
cplt.plot_hline(ax=ax3,
                y=year_separte_arr['val'],
                lw=2,
                color='red',
                label='sync. years (top 25%)')
cplt.plot_hline(ax=ax3,
                y=year_separte_arr['val_'],
                lw=2,
                color='green',
                label='few sync. (lowest 25%)')
# cplt.plot_hline(ax=ax3,
#                 y=12,
#                 lw=2,
#                 color='black',
#                 label='Average')
cplt.set_legend(ax=ax3,
                fig=im['fig'],
                loc='outside',
                box_loc=(-.05, 0.0),
                ncol_legend=3
                )

# Plot Weekly distributions
lagged_sync_ts = tps_dict['lagged_index']
jjas_weeks = np.unique(tu.get_time_count_number(tps=lagged_sync_ts,
                                                counter='week'))

jjas_week_labels = tu.get_week_dates(jjas_weeks.data)
week_numbers = tsa.count_tps_occ(
    tps_arr=[tps_sync_jjas.time],
    counter='week',
    count_arr=jjas_weeks,
    rel_freq=False, norm_fac=1,)

im_bars = cplt.plot_xy(
    ax=ax4,
    x_arr=jjas_week_labels,
    y_arr=[week_numbers/40],
    plot_type='bar',
    # label_arr=['Most Synchronous Days'],
    title=rf'Distribution of synchronizations wrt week of the year',
    # xlabel='Month',
    ylabel='No. most sync. days',
    set_legend=True,
    loc='upper right',)


savepath = (
    plot_dir
    + f"/paper_plots/msl_ts.png"
)
cplt.save_fig(savepath)



# %%
# Load MSE data
reload(mse)

nc_files_q = []
nc_files_t = []
nc_files_z = []

plevels = np.arange(200, 1050, 200)
plevels = np.arange(100, 1050, 100)

for plevel in plevels:
    dataset_file_q = data_dir + \
        f"/climate_data/2.5/era5_q_{2.5}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
    dataset_file_t = data_dir + \
        f"/climate_data/2.5/era5_t_{2.5}_{plevel}_ds.nc"
    nc_files_t.append(dataset_file_t)
    dataset_file_z = data_dir + \
        f"/climate_data/2.5/era5_z_{2.5}_{plevel}_ds.nc"
    nc_files_z.append(dataset_file_z)

# %%
reload(mp)
ds_q = mp.MultiPressureLevelDataset(data_nc=nc_files_q,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    month_range=['Jun', 'Sep'],
                                    lon_range=lon_range_cut,
                                    lat_range=lat_range_cut,
                                    plevels=plevels,
                                    )
# %%
ds_t = mp.MultiPressureLevelDataset(data_nc=nc_files_t,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    month_range=['Jun', 'Sep'],
                                    lon_range=lon_range_cut,
                                    lat_range=lat_range_cut,
                                    plevels=plevels,
                                    metpy_unit='K'
                                    )

# %%
reload(mp)
ds_z = mp.MultiPressureLevelDataset(data_nc=nc_files_z,
                                    plevels=plevels,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    month_range=['Jun', 'Sep'],
                                    lon_range=lon_range_cut,
                                    lat_range=lat_range_cut,
                                    metpy_unit='m',
                                    )
# %%
# IWF
reload(wds)
dataset_file_ewf = data_dir + \
    f"/climate_data/2.5/era5_ewvf_{2.5}_ds.nc"
dataset_file_nwf = data_dir + \
    f"/climate_data/2.5/era5_nwvf_{2.5}_ds.nc"

ds_ivf = wds.Wind_Dataset(data_nc_u=dataset_file_ewf,
                          data_nc_v=dataset_file_nwf,
                          can=True,
                          an_types=['month', 'JJAS'],
                          u_name='ewvf',
                          v_name='nwvf',
                          month_range=['Jun', 'Sep'],
                          lon_range=lon_range_cut,
                          lat_range=lat_range_cut,
                          compute_ws=True,
                          )
# %%
# Compute relative humidity
reload(mut)
rh = mut.specific_humidity_to_relative_humidity(
    pressure=ds_q.ds['lev'],
    temperature=ds_t.ds['t'],
    specific_humidity=ds_q.ds['q'],
    percentage=False
)
rh = tu.compute_anomalies_ds(rh, )
# %%
# Compute potential temperature
reload(mut)
pt = mut.potential_temperature(
    pressure=ds_t.ds['t'].lev,
    temperature=ds_t.ds['t']
)
# %%
reload(mut)
cross_data = gut.merge_datasets(rh, pt)

# %%
# OLR data
reload(bds)
dataset_file = data_dir + \
    f"climate_data/1/era5_ttr_{1}_ds.nc"

ds_olr = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         an_types=['dayofyear', 'month', 'JJAS'],
                         lon_range=lon_range_cut,
                         lat_range=lat_range_cut,
                         month_range=['Jun', 'Sep'],
                         )
# %%
dataset_file = data_dir + \
    f"/climate_data/2.5/era5_sst_{2.5}_ds.nc"

ds_sst = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         an_types=['JJAS', 'month'],
                         month_range=['Jun', 'Sep'],
                         #  lon_range=lon_range_cut,
                         #  lat_range=lat_range_cut,
                         )

# %%
# precipitation
reload(bds)
dataset_file = data_dir + \
    f"climate_data/1/mswep_pr_{1}_ds.nc"

ds_pr = bds.BaseDataset(data_nc=dataset_file,
                        can=True,
                        an_types=['dayofyear', 'month', 'JJAS'],
                        lon_range=lon_range_cut,
                        lat_range=lat_range_cut,
                        )
# %%
# Load wind fields
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []
levs = [200, 400, 600, 800
        ]
levs = np.arange(100, 1050, 100)
for lev in levs:
    dataset_file_u = data_dir + \
        f"/climate_data/2.5/era5_u_{2.5}_{lev}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/2.5/era5_v_{2.5}_{lev}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = data_dir + \
        f"/climate_data/2.5/era5_w_{2.5}_{lev}_ds.nc"
    nc_files_w.append(dataset_file_w)

reload(wds)
ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           data_nc_w=nc_files_w,
                           plevels=levs,
                           can=True,
                           an_types=['month', 'JJAS'],
                           month_range=['Jun', 'Sep'],
                           init_mask=False,
                           lon_range=lon_range_cut,
                           lat_range=lat_range_cut,
                           )
# %%
an_type = 'month'
u_var_type = f'u_an_{an_type}'
v_var_type = f'v_an_{an_type}'
w_var_type = f'OMEGA_an_{an_type}'
times = ds_wind.ds.time
cross_data_wind = ds_wind.ds[[u_var_type,
                              v_var_type,
                              w_var_type]]

# %%
# Orography
oro_res = 0.25
dataset_file_orograph = data_dir + \
    f"/climate_data/{oro_res}/era5_orography_{oro_res}_ds.nc"
ds_orograph = bds.BaseDataset(data_nc=dataset_file_orograph,
                              lon_range=lon_range_cut,
                              lat_range=lat_range_cut,
                              grid_step=oro_res,
                              )

# %%
reload(bds)
lev = 200
dataset_file_u = data_dir + \
    f"/climate_data/1/era5_u_{1}_{lev}_ds.nc"

u200 = bds.BaseDataset(data_nc=dataset_file_u,
                       can=True,
                       an_types=['dayofyear', 'month'],
                       lat_range=[0, 90],  # Only northern hemisphere
                       month_range=['Jun', 'Sep'],
                       init_mask=False)
# %%
reload(tej)
an_type = 'month'
var_type = f'an_{an_type}'
# var_type = 'u'
tej_index = tej.get_tej_index(u200.ds[var_type])
tej_dict = tej.get_tej_strength(u200=u200.ds[var_type],
                                tej_val=3,
                                start_month='Jun',
                                end_month='Sep',
                                northward_extension=False,
                                get_index=False,)


# %%
reload(tu)
reload(bs)
index_def = 'Kikuchi'
bsiso_index = bs.get_bsiso_index(
    start_month='Jun', end_month='Sep', index_def=index_def)
# %%
# create time points dictionary
reload(tu)
reload(gut)
reload(cplt)
reload(tsa)
reload(sut)
reload(bs)
reload(mei)
mei_index = mei.get_mei_index(start_month='Jun',
                              end_month='Sep')
enso_types = mei.get_enso_types(start_month='Jun',
                                end_month='Sep')
nina_tps = enso_types['La Nina']

comp_th = 0.9
lag = 12
include_ts = False
if include_ts:
    include_region = 'SZ'
    savepath = plot_dir + \
        f"synchronization/synchronization_lag_{lag}_include_{include_region}_{comp_th}_ts_dict.npy"
else:
    num_lags = 5
    savepath = plot_dir + \
        f"synchronization/synchronization_lags_{num_lags}_{comp_th}_ts_dict.npy"

lagged_sync_dict = fut.load_np_dict(savepath)
sregion = 'NI_SZ'
# sregion = 'SZ_NI'

# Synchronous and not synchronous days
tps_sync = lagged_sync_dict[sregion][lag]['tps_active']
tps_sync_ja = lagged_sync_dict[sregion][lag]['tps_sync_ja']
tps_sync_jjas = lagged_sync_dict[sregion][lag]['tps_sync_jjas']

break_years = lagged_sync_dict[sregion][lag]['break_years']
lagged_active_years = lagged_sync_dict[sregion][lag]['active_years']

background_tps = tu.get_periods_tps(tps=tps_sync,
                                    end=-10,
                                    start=-0)

nina_tps = enso_types['La Nina'].time
nino_tps = enso_types['El Nino'].time

lagged_index = lagged_sync_dict[sregion][lag]['sync_ts']
tr_jjas = tu.get_dates_of_time_range(['1979-01-01',
                                      '2021-01-01'],
                                     start_month='Jun',
                                     end_month='Sep')

tr_ja = tu.get_dates_of_time_range(['1979-01-01',
                                    '2021-01-01'],
                                   start_month='Jul',
                                   end_month='Aug')

phase_num = 6
tps_dict = {
    'lagged_index': lagged_index,
    # 'tej_index': tej_dict['index']['tej'],
    # 'cgti_index': cgti_dict['index']['cgti'],
    'tps_sync': tps_sync,
    'tps_sync_ja': tps_sync_ja,
    'tps_sync_jjas': tps_sync_jjas,
    'background active': background_tps,
    'tej_pos': tej_dict['enhanced'],
    'tej_neg': tej_dict['reduced'],
    'nina': nina_tps,
    'nino': nino_tps,
    'JJAS': tr_jjas,
    'JA': tr_ja,
    'active_ja': lagged_sync_dict[sregion][lag]['active_years'],
    'break_years': lagged_sync_dict[sregion][lag]['break_years'],
}
# %%
# Cluster time points
method = 'kmeans'
num_lags = 10
n_eofs = 1
n_cluster = 2
variables = [
    # 'olr',
    'u200',
    'v200',
    'rh400',
    # 'rh700',
    'w500'
]
prefix_cluster = f"latent_mvpca_{variables}_eof{n_eofs}_n{n_cluster}_{method}_lags_{num_lags}_{comp_th}"

savepath = plot_dir + \
    f"latent_clustering/{prefix_cluster}.npy"
latent_cluster = fut.load_np_dict(sp=savepath)



# %%
# Plot the propagation of the synchronization
# OLR + vertical velocity plots
reload(cplt)
reload(tu)
lon_range = [-30, 140]
lat_range = [-20, 60]

an_type = 'month'
lev = 400
var_type_olr = f'an_{an_type}'


label_olr = f'An. OLR (wrt {an_type}) [W/m²]'
vmax_olr = 2.e1
vmin_olr = -vmax_olr

w_var_type = f'OMEGA_an_JJAS'
vmin_w = -2
vmax_w = -vmin_w
step = 4
steps = np.arange(0, 13, step)
ncols = n_cluster
nrows = len(steps)

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            hspace=0.,
                            wspace=0.4,
                            projection='PlateCarree',
                            lat_range=lat_range,
                            lon_range=lon_range,
                            )
for idx, group in enumerate(latent_cluster['keys']):
    tps = latent_cluster[group]
    print(f"Group {group} has {len(tps)} time points")
    for d, start in enumerate(steps):

        sel_tps = tu.add_time_step_tps(tps, time_step=start)

        mean_data, sig_data = tu.get_mean_tps(ds_olr.ds[var_type_olr],
                                              tps=sel_tps,
                                              corr_type=None)
        im_prop = cplt.plot_map(mean_data*sig_data,
                                ax=im['ax'][ncols*d + idx],
                                title=f'{group} ({len(tps)} samples)' if d == 0 else None,
                                y_title=1.3,
                                vertical_title=f'Day {start}' if idx == 0 else None,
                                plot_type='contourf',
                                cmap='RdBu_r',
                                levels=12,
                                label=label_olr,
                                vmin=vmin_olr, vmax=vmax_olr,
                                tick_step=2,
                                centercolor='white',
                                extend='both',
                                orientation='vertical',
                                )

        mean_data, sig_data = tu.get_mean_tps(ds_wind.ds[w_var_type].sel(lev=lev),
                                              tps=sel_tps,
                                              corr_type=None)
        cplt.plot_map(mean_data*sig_data*100,  # convert to Pa/s,
                      ax=im_prop['ax'],
                      plot_type='contour',
                      color='solid_dashed',
                      levels=2,
                      vmin=vmin_w, vmax=vmax_w,
                      lw=1,
                      clabel=True,
                      )

savepath = plot_dir +\
    f"paper_plots/propagation_olr_{prefix_cluster}.png"
cplt.save_fig(savepath)

# %% U V Z 200
reload(cplt)
groups = ['Strong Propagation',
          'Weak Propagation',
          'Dry Years']

tps_arr = [
    latent_cluster[groups[0]],
    latent_cluster[groups[1]],
    tps_dict['break_years'],
]

an_type = 'month'
label_sst = f'SST Anomalies (wrt {an_type}) [K]'
vmin_sst = -1
vmax_sst = -vmin_sst
lev = 200
nrows = 3
ncols = len(tps_arr)
split_1980 = True
im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2,
                            hspace=0.6,
                            dateline_arr=[
                                #    True, True,   True,
                                False, False, False,
                                False, False, False,
                                False, False, False
                            ],
                            enumerate_subplots=True)

for idx, this_tps in enumerate(tps_arr):
    if split_1980:
        this_tps = tu.get_time_range_data(this_tps,
                                          time_range=['1987-01-01', '2020-12-31'])


    mean_tps_u, sig_u = tu.get_mean_tps(ds_wind.ds[f'u_an_{an_type}'].sel(lev=lev),
                                        this_tps.time)
    vmax = 6
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_u*sig_u,
                            ax=im['ax'][0*ncols + idx],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f'{groups[idx]} (Day 0)' if idx <= 1 else f'{groups[idx]} (JA)',
                            vertical_title=f"U200 Anomalies [m/s]" if idx == 0 else None,
                            label=rf'U-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            lon_range=[-20, 180],
                            lat_range=[-20, 70],
                            )

    mean_tps_v, sig_v = tu.get_mean_tps(ds_wind.ds[f'v_an_{an_type}'].sel(lev=lev),
                                        this_tps.time)
    vmax = 5
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_v*sig_v,
                            ax=im['ax'][1*ncols + idx],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            # title=f'{group}',
                            vertical_title=f"V200 Anomalies [m/s]" if idx == 0 else None,
                            label=rf'V-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            lon_range=[-20, 180],
                            lat_range=[-20, 70],
                            )

    mean_tps, sig_z = tu.get_mean_tps(ds_z.ds[f'an_{an_type}'].sel(lev=lev),
                                      this_tps.time)

    sig_uv = sut.sig_multiple_field(sig_mask_x=sig_u,
                                    sig_mask_y=sig_v)
    vmax = .7e3
    vmin = -vmax
    im_comp = cplt.plot_map(mean_tps,
                            ax=im['ax'][2*ncols + idx],
                            plot_type='contourf',
                            cmap='RdYlBu_r',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            # title=f'{group}',
                            vertical_title=r"GP200 Anomalies [m$^2$/s$^2$]" if idx == 0 else None,
                            label=rf'Anomalies GP (wrt {an_type}) [$m^2/s^2$]',
                            lon_range=[-20, 180],
                            lat_range=[-20, 70],
                            )
    dict_w = cplt.plot_wind_field(ax=im_comp['ax'],
                                  u=mean_tps_u*sig_uv,
                                  v=mean_tps_v*sig_uv,
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=50,
                                  steps=2,
                                  key_length=1,
                                  )

savepath = plot_dir + \
    f"paper_plots/background_uvz{lev}_{an_type}_{prefix_cluster}.png"
cplt.save_fig(savepath=savepath, fig=im['fig'])

# %% ################ Vertical Cross Sections ####################
region = 'NI'
if region == 'NI':
    # NI
    c_lon_range = [60, 100]  # This is the plotting range for zonal
    c_lat_range = [20, 30]
    s_lon_range = [70, 80]
    s_lat_range = [0, 40]  # This is the plotting range for meridional
elif region == 'SZ':
    # SZ
    c_lon_range = [-25, 60]  # This is the plotting range for zonal
    c_lat_range = [10, 16]
    s_lon_range = [0, 15]
    s_lat_range = [-0, 30]  # This is the plotting range for meridional
else:
    raise ValueError('Region not found')
wind_data_lon = mut.vertical_cross_section(cross_data_wind,
                                           lon_range=c_lon_range,
                                           lat_range=c_lat_range,
                                           )
wind_data_lat = mut.vertical_cross_section(cross_data_wind,
                                           lon_range=s_lon_range,
                                           lat_range=s_lat_range,
                                           )
rh_data_lon = mut.vertical_cross_section(cross_data,
                                         lon_range=c_lon_range,
                                         lat_range=c_lat_range,
                                         )
rh_data_lat = mut.vertical_cross_section(cross_data,
                                         lon_range=s_lon_range,
                                         lat_range=s_lat_range,
                                         )

oro_data_lon = sput.cut_map(ds_orograph.ds['z'],
                            lon_range=c_lon_range,
                            lat_range=c_lat_range,
                            dateline=False)
oro_data_lat = sput.cut_map(ds_orograph.ds['z'],
                            lon_range=s_lon_range,
                            lat_range=s_lat_range,
                            dateline=False)
oro_data_lon = oro_data_lon.max(dim='lat')
oro_data_lat = oro_data_lat.max(dim='lon')

# Pr data
pr_data_lon = sput.cut_map(ds_pr.ds,
                           lon_range=c_lon_range,
                           lat_range=c_lat_range)
pr_data_lat = sput.cut_map(ds_pr.ds,
                           lon_range=s_lon_range,
                           lat_range=s_lat_range,
                           )
# %% vertical cross section plots
# overturning ciruclation plots for NI/SZ
reload(cplt)
reload(tu)
group = 0
tps_0 = latent_cluster[group]
if region == 'NI':
    vmin_pr = -2.5
    vmax_pr = 9.5
else:
    vmin_pr = -1.
    vmax_pr = 9.
vmin_vert = 0.
vmax_vert = 25

plot_pt = False
background = False
an_type = 'month'
var_type_wind_u = f'u_an_{an_type}'
var_type_wind_v = f'v_an_{an_type}'
w_var_type = f'OMEGA_an_{an_type}'

an_type = 'JJAS'
var_type_rh = f'rh_an_{an_type}'
var_type_pr = f'an_{an_type}'
label_u = rf'RH Anomalies (wrt month) [%]'
label_v = rf'RH Anomalies (wrt month) [%]'
label_pr = 'Pr. an. [mm/day]'

y_title = 1.4
key_loc = (0.95, 1.35)
pr_color = 'tab:blue'
yticks = np.arange(0, 1050, 100)
start = 11
steps = [0, 1]
nrows = len(steps)
ncols = 2


im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            wspace=0.2,
                            hspace=0.55,
                            figsize=(7*ncols, 6*nrows),
                            pos_y=1.35,
                            )
for idx, step in enumerate(steps):

    this_tps = tu.get_tps_range(tps_0,
                                start=start if idx == 1 else 0,
                                time_step=step)
    rh_cross_lon, _ = tu.get_mean_tps(
        da=rh_data_lon[var_type_rh], tps=this_tps,
    )
    rh_cross_lat, _ = tu.get_mean_tps(
        da=rh_data_lat[var_type_rh], tps=this_tps,
    )
    lons = rh_cross_lon.lon[0]
    lats = rh_cross_lat.lat[0]
    plevels = rh_cross_lon.isobaric

    # Wind fields in zonal direction
    k_data_u, sig_u = tu.get_mean_tps(
        da=wind_data_lon[u_var_type], tps=this_tps)

    k_data_uw, sig_uw = tu.get_mean_tps(
        da=wind_data_lon[w_var_type], tps=this_tps)

    # Meridional data
    k_data_v, sig_v = tu.get_mean_tps(
        da=wind_data_lat[v_var_type], tps=this_tps)

    k_data_vw, sig_vw = tu.get_mean_tps(
        da=wind_data_lat[w_var_type], tps=this_tps)

    # Orography
    ground_level = 1024
    oro_lon = oro_data_lon.data/200*-1 + ground_level
    oro_lat = oro_data_lat.data/200*-1 + ground_level
    # Precipitation
    pr_zonal = tu.get_mean_tps(da=pr_data_lon,
                               sig_test=False,
                               tps=this_tps,
                               )
    pr_zonal = sput.horizontal_average(pr_zonal, dim='lat',
                                       average_type='mean' if region == 'NI' else 'max')
    pr_zonal = pr_zonal * 0.7 if step == 0 and region == 'SZ' else pr_zonal
    pr_meridional = tu.get_mean_tps(da=pr_data_lat,
                                    tps=this_tps,
                                    sig_test=False,
                                    )
    pr_meridional = sput.horizontal_average(pr_meridional, dim='lon',
                                            average_type='mean' if region == 'NI' else 'max')
    pr_meridional = pr_meridional * 0.7 if step == 0 else pr_meridional

    # Zonal circulation
    day_str = 0 if idx == 0 else start + step
    h_im_u = cplt.plot_2D(x=lons, y=plevels,
                          ax=im['ax'][idx*ncols],
                          z=rh_cross_lon*100,
                          levels=16,
                          vertical_title=f'Day {day_str}',
                          title='Zonal Circulation' if idx == 0 else None,
                          cmap='YlGnBu',
                          #   centercolor='white',
                          plot_type='contourf',
                          extend='both',
                          vmin=vmin_vert, vmax=vmax_vert,
                          xlabel='Longitude [degree]' if idx == 1 else None,
                          ylabel='Pressure Level [hPa]',
                          flip_y=True,
                          label=label_v if idx == nrows - 1 else None,
                          orientation='horizontal',
                          pad=-5,
                          x_title_offset=-0.25,
                          y_title=y_title,
                          top_ticks=True,
                          #   ysymlog=True,
                          #   yticks=yticks,
                          ylim=[50, 1000],
                          round_dec=0,
                          )

    cplt.plot_xy(
        ax=h_im_u['ax'],
        x_arr=[oro_data_lon.lon],
        y_arr=[oro_lon],
        ls_arr=['-'],
        mk_arr=[''],
        lw_arr=[3],
        color='grey',
        zorder=20,
        filled=True,
        offset_fill=ground_level,
    )
    ax_pos = h_im_u['ax'].get_position()
    # Calculate position for the new axis below the existing subplot
    new_ax_height = 0.1  # Height of the new axis
    new_ax_y = ax_pos.y1  # + 0.02  # Adjust 0.02 for padding

    ax_pr = im['fig'].add_axes(
        [ax_pos.x0, new_ax_y, ax_pos.width, new_ax_height])
    cplt.plot_xy(
        ax=ax_pr,
        x_arr=[pr_zonal.lon],
        y_arr=[pr_zonal[var_type_pr].data],
        set_axis=True,
        set_twinx=False,
        ylabel=label_pr,
        ylabel_pos=(-0.14, 0.45),
        lw_arr=[3],
        ylabel_color=pr_color,
        color_arr=[pr_color],
        ylim=[vmin_pr, vmax_pr],
        xlim=[c_lon_range[0], c_lon_range[1]],
        set_ticks=True,
        xticklabels=[],
        set_grid=True,
    )

    sig_uw = sut.sig_multiple_field(sig_mask_x=sig_u,
                                    sig_mask_y=sig_uw)
    dict_w = cplt.plot_wind_field(
        ax=h_im_u['ax'],
        u=k_data_u.where(sig_uw, 0),
        v=k_data_uw.where(sig_uw, 0)*100,
        x_vals=lons,
        y_vals=k_data_uw.isobaric,
        x_steps=3,
        transform=False,
        scale=40,
        width=0.004,
        key_length=1,
        wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
        key_loc=key_loc
    )

    # Meridional circulation
    h_im_v = cplt.plot_2D(x=lats, y=plevels,
                          ax=im['ax'][idx*ncols + 1],
                          z=rh_cross_lat*100,
                          levels=24,
                          title='Meridional Circulation' if idx == 0 else None,
                          cmap='YlGnBu',
                          #   centercolor='white',
                          plot_type='contourf',
                          extend='both',
                          vmin=vmin_vert, vmax=vmax_vert,
                          xlabel='Latitude [degree]' if idx == 1 else None,
                          #   ylabel='Pressure Level [hPa]',
                          flip_y=True,
                          label=label_u if idx == nrows - \
                          1 else None,
                          orientation='horizontal',
                          pad=-5,
                          x_title_offset=-0.25,
                          y_title=y_title,
                          ylim=[50, 1000],
                          round_dec=0,
                          tick_step=3,
                          top_ticks=True,
                          )
    cplt.plot_xy(
        ax=h_im_v['ax'],
        x_arr=[oro_data_lat.lat],
        y_arr=[oro_lat],
        ls_arr=['-'],
        mk_arr=[''],
        lw_arr=[3],
        color='grey',
        zorder=20,
        filled=True,
        offset_fill=ground_level,
    )
    ax_pos = h_im_v['ax'].get_position()
    # Calculate position for the new axis below the existing subplot
    new_ax_height = 0.1  # Height of the new axis
    new_ax_y = ax_pos.y1  # + 0.02  # Adjust 0.02 for padding

    ax_pr = im['fig'].add_axes(
        [ax_pos.x0, new_ax_y, ax_pos.width, new_ax_height])
    cplt.plot_xy(
        ax=ax_pr,
        x_arr=[pr_meridional.lat],
        y_arr=[pr_meridional[var_type_pr].data],
        set_axis=True,
        set_twinx=False,
        # ylabel=label_pr,
        lw_arr=[3],
        ylabel_color=pr_color,
        color_arr=[pr_color],
        ylim=[vmin_pr, vmax_pr],
        xlim=[s_lat_range[0], s_lat_range[1]],
        set_ticks=False,
        set_grid=True,
    )

    sig_vw = sut.sig_multiple_field(sig_mask_x=sig_v,
                                    sig_mask_y=sig_vw)
    dict_w = cplt.plot_wind_field(
        ax=h_im_v['ax'],
        u=k_data_v.where(sig_vw, 0),
        v=k_data_vw.where(sig_vw, 0)*100,
        x_vals=lats,
        y_vals=k_data_v.isobaric,
        x_steps=3,
        transform=False,
        scale=30,
        width=0.005,
        key_length=1,
        wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
        key_loc=key_loc
    )


savepath = plot_dir +\
    f"clustering/{region}_group{group}_rh_cross_section_{prefix_cluster}.png"
cplt.save_fig(savepath)

# %%  Figure 6
# NI convection conditions:
reload(cplt)
group = 'Strong Propagation'
if region == 'NI':
    this_lon_range = [40, 130]
    this_lat_range = [0, 45]
    this_tps = latent_cluster[group]
elif region == 'SZ':
    this_lon_range = [-20, 70]
    this_lat_range = [-10, 40]
    this_tps = tu.add_time_step_tps(latent_cluster[group],
                                    time_step=11)

if region == 'NI':
    vmin_pr = -2.5
    vmax_pr = 9.5
else:
    vmin_pr = -1.
    vmax_pr = 9.

an_type = 'month'
var_type_wind_u = f'u_an_{an_type}'
var_type_wind_v = f'v_an_{an_type}'
w_var_type = f'OMEGA_an_{an_type}'

label_u = rf'RH Anomalies (wrt month) [%]'
label_v = rf'RH Anomalies (wrt month) [%]'
label_pr = r'Pr. an. $[\frac{mm}{day}]$'

an_type = 'JJAS'
var_type_pr = f'an_{an_type}'
var_type_rh = f'rh_an_{an_type}'

y_title = 1.5
pr_ax_height = 0.1
key_loc = (1.2, 1.42)
pr_color = 'tab:blue'
nrows = 2
ncols = 2
im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            wspace=0.35,
                            hspace=1.,
                            figsize=(12, 10),
                            ratios_w=[2., 1.2, ],
                            proj_arr=['PlateCarree', None, None, None],
                            pos_y=y_title,
                            pos_x=-0.25
                            )
q_oro = ds_orograph.ds['z'].quantile(q=0.95)
himalayan_mask = xr.where(ds_orograph.ds['z'] < q_oro, np.nan, 1)
level = 400

label_olr = f'OLR Anomalies (wrt month) [W/m²]'
vmax_olr = 2.e1
vmin_olr = -vmax_olr
mean_data, sig_data = tu.get_mean_tps(ds_olr.ds['an_month'],
                                      tps=this_tps)
vmin_w = -.6e-1*100
vmax_w = -vmin_w

vmin_vert = 0
vmax_vert = 25

im_map = cplt.plot_map(mean_data,
                       ax=im['ax'][0],
                       title='Convective anomalies (Day 0)',
                       y_title=y_title,
                       plot_type='contourf',
                       cmap='RdBu_r',
                       levels=20,
                       vmin=vmin_olr, vmax=vmax_olr,
                       tick_step=2,
                       label=f'{label_olr}',
                       centercolor='white',
                       pad=-1,
                       lon_range=this_lon_range,
                       lat_range=this_lat_range,
                       plot_grid=True,
                       )
mean_data, sig_data = tu.get_mean_tps(ds_wind.ds[w_var_type].sel(lev=level),
                                      tps=this_tps)
cplt.plot_map(mean_data*100,
              ax=im_map['ax'],
              plot_type='contour',
              color='solid_dashed',
              color_contour=['orange', 'purple'],
              plt_grid=False,
              levels=4,
              lw=2,
              #   clabel=True,
              vmin=vmin_w, vmax=vmax_w,
              )
im_ax = cplt.plot_map(himalayan_mask,
                      ax=im['ax'][0],
                      plot_type='contourf',
                      color='grey',
                      alpha=0.6,
                      levels=2,
                      vmin=0, vmax=1,
                      tick_step=2,
                      zorder=1,
                      )
for (tlon, tlat) in [[c_lon_range, c_lat_range],
                     [s_lon_range, s_lat_range]]:
    cplt.plot_rectangle(ax=im_map['ax'],
                        lon_range=tlon,
                        lat_range=tlat,
                        color='black',
                        lw=2,
                        ls='--',)


rh_cross_lon, _ = tu.get_mean_tps(
    da=rh_data_lon[var_type_rh], tps=this_tps,
)
rh_cross_lat, _ = tu.get_mean_tps(
    da=rh_data_lat[var_type_rh], tps=this_tps,
)

lons = rh_cross_lon.lon[0]
lats = rh_cross_lat.lat[0]
plevels = rh_cross_lon.isobaric

# Wind fields in zonal direction
k_data_u, sig_u = tu.get_mean_tps(
    da=wind_data_lon[u_var_type], tps=this_tps)

k_data_uw, sig_uw = tu.get_mean_tps(
    da=wind_data_lon[w_var_type], tps=this_tps)

# Meridional data
k_data_v, sig_v = tu.get_mean_tps(
    da=wind_data_lat[v_var_type], tps=this_tps)

k_data_vw, sig_vw = tu.get_mean_tps(
    da=wind_data_lat[w_var_type], tps=this_tps)

# Orography
ground_level = 1024
oro_lon = oro_data_lon.data/300*-1 + ground_level
oro_lat = oro_data_lat.data/300*-1 + ground_level
# Precipitation
pr_zonal = tu.get_mean_tps(da=pr_data_lon,
                           tps=this_tps,
                           sig_test=False,
                           )
pr_zonal = sput.horizontal_average(pr_zonal, dim='lat',
                                   average_type='median' if region == 'NI' else 'max')
pr_meridional = tu.get_mean_tps(da=pr_data_lat,
                                tps=this_tps,
                                sig_test=False,
                                )
pr_meridional = sput.horizontal_average(pr_meridional, dim='lon',
                                        average_type='median' if region == 'NI' else 'max')

# Zonal circulation
h_im_u = cplt.plot_2D(x=lons, y=plevels,
                      ax=im['ax'][1],
                      z=rh_cross_lon*100,
                      levels=21,
                      title='Zonal Circulation (Day 0)',
                      cmap='YlGnBu',
                      #   centercolor='white',
                      plot_type='contourf',
                      extend='both',
                      vmin=vmin_vert, vmax=vmax_vert,
                      xlabel='Longitude [degree]',
                      ylabel='Pressure Level [hPa]',
                      flip_y=True,
                      label=label_v,
                      orientation='vertical',
                      #   pad=-5,
                      x_title_offset=-0.25,
                      y_title=y_title,
                      top_ticks=True,
                      #   ysymlog=True,
                      #   yticks=yticks,
                      ylim=[50, 1000],
                      round_dec=0,
                      tick_step=3,
                      )

cplt.plot_xy(
    ax=h_im_u['ax'],
    x_arr=[oro_data_lon.lon],
    y_arr=[oro_lon],
    ls_arr=['-'],
    mk_arr=[''],
    lw_arr=[3],
    color='grey',
    zorder=20,
    filled=True,
    offset_fill=ground_level,
)
ax_pos = h_im_u['ax'].get_position()
# Calculate position for the new axis below the existing subplot
new_ax_height = pr_ax_height  # Height of the new axis
new_ax_y = ax_pos.y1  # + 0.02  # Adjust 0.02 for padding

ax_pr = im['fig'].add_axes(
    [ax_pos.x0, new_ax_y, ax_pos.width, new_ax_height])
cplt.plot_xy(
    ax=ax_pr,
    x_arr=[pr_zonal.lon],
    y_arr=[pr_zonal[var_type_pr].data],
    set_axis=True,
    set_twinx=False,
    ylabel=label_pr,
    ylabel_pos=(-0.2, 0.5),
    lw_arr=[3],
    ylabel_color=pr_color,
    color_arr=[pr_color],
    ylim=[vmin_pr, vmax_pr],
    xlim=[c_lon_range[0], c_lon_range[1]],
    set_ticks=True,
    xticklabels=[],
    set_grid=True,
)

sig_uw = sut.sig_multiple_field(sig_mask_x=sig_u,
                                sig_mask_y=sig_uw)
dict_w = cplt.plot_wind_field(
    ax=h_im_u['ax'],
    u=k_data_u.where(sig_uw, 0),
    v=k_data_uw.where(sig_uw, 0)*100,
    x_vals=lons,
    y_vals=k_data_uw.isobaric,
    x_steps=3,
    transform=False,
    scale=40,
    width=0.004,
    key_length=1,
    wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
    key_loc=key_loc
)

# Meridional circulation
h_im_v = cplt.plot_2D(x=lats, y=plevels,
                      ax=im['ax'][3],
                      z=rh_cross_lat*100,
                      levels=21,
                      title='Meridional Circulation (Day 0)',
                      cmap='YlGnBu',
                      #   centercolor='white',
                      plot_type='contourf',
                      extend='both',
                      vmin=vmin_vert, vmax=vmax_vert,
                      xlabel='Latitude [degree]',
                      ylabel='Pressure Level [hPa]',
                      flip_y=True,
                      label=label_u,
                      orientation='vertical',
                      #   pad=-5,
                      x_title_offset=-0.25,
                      y_title=y_title,
                      ylim=[50, 1000],
                      round_dec=0,
                      tick_step=3
                      )
cplt.plot_xy(
    ax=h_im_v['ax'],
    x_arr=[oro_data_lat.lat],
    y_arr=[oro_lat],
    ls_arr=['-'],
    mk_arr=[''],
    lw_arr=[3],
    color='grey',
    zorder=20,
    filled=True,
    offset_fill=ground_level,
)
ax_pos = h_im_v['ax'].get_position()
# Calculate position for the new axis below the existing subplot
new_ax_height = pr_ax_height  # Height of the new axis
new_ax_y = ax_pos.y1  # + 0.02  # Adjust 0.02 for padding

ax_pr = im['fig'].add_axes(
    [ax_pos.x0, new_ax_y, ax_pos.width, new_ax_height])
cplt.plot_xy(
    ax=ax_pr,
    x_arr=[pr_meridional.lat],
    y_arr=[pr_meridional[var_type_pr].data],
    set_axis=True,
    set_twinx=False,
    ylabel=label_pr,
    ylabel_pos=(-0.2, 0.5),
    lw_arr=[3],
    ylabel_color=pr_color,
    color_arr=[pr_color],
    ylim=[vmin_pr, vmax_pr],
    xlim=[s_lat_range[0], s_lat_range[1]],
    set_ticks=False,
    set_grid=True,
)

sig_vw = sut.sig_multiple_field(sig_mask_x=sig_v,
                                sig_mask_y=sig_vw)
dict_w = cplt.plot_wind_field(
    ax=h_im_v['ax'],
    u=k_data_v.where(sig_vw, 0),
    v=k_data_vw.where(sig_vw, 0)*100,
    x_vals=lats,
    y_vals=k_data_v.isobaric,
    x_steps=3,
    transform=False,
    scale=30,
    width=0.005,
    key_length=1,
    wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
    key_loc=key_loc
)

reload(tsa)
reload(sut)
reload(bs)

time_range = ['1981-01-01', '2019-12-31']
sregion = 'NI_SZ'
lag = 15

act_th = 1
break_th = .9
index_def = 'Kikuchi'
phase_vals = np.arange(1, 9, 1)

nrows = len(enso_types)
ncols = len(tej_dict)

# Synchronous and not synchronous days
tps_sync = tps_dict['tps_sync_jjas']

# BSISO Index Phase Active
bsiso_index = bs.get_bsiso_index(time_range=time_range,
                                 start_month='Jun',
                                 end_month='Sep',
                                 index_def=index_def)
times = bsiso_index.time
bsiso_phase = bsiso_index['BSISO-phase']
null_model_uniform = len(tps_sync) / len(times)
null_model = sut.count_occ([bsiso_phase.data],
                           count_arr=phase_vals) / \
    len(bsiso_phase.time)*len(phase_vals) * null_model_uniform

ampl = bsiso_index['BSISO-ampl']

phase_sync = tu.get_sel_tps_ds(bsiso_phase, tps=tps_sync)

# Get all active/break days per phase
act_days_phase = xr.where(ampl >= act_th, bsiso_phase,
                          np.nan).dropna(dim='time')
break_days_phase = xr.where(
    ampl < break_th, bsiso_phase, np.nan).dropna(dim='time')

p_s_1_act = tsa.get_cond_occ(
    tps=tps_sync, cond=act_days_phase, counter=phase_vals)

p_s_1_break = tsa.get_cond_occ(
    tps=tps_sync, cond=break_days_phase, counter=phase_vals)

# include tej
tej_act_days = tu.get_sel_tps_ds(ds=act_days_phase, tps=tps_dict['tej_pos'])
tej_break_days = tu.get_sel_tps_ds(
    ds=break_days_phase, tps=tps_dict['tej_pos'])

# include ENSO
enso_act_days = tu.get_sel_tps_ds(ds=act_days_phase,
                                  tps=tps_dict['nina'])
enso_break_days = tu.get_sel_tps_ds(ds=break_days_phase, tps=tps_dict['nina'])

# get days that are ENSO + tej + active/break
tej_act_enso = tu.get_sel_tps_ds(
    tej_act_days, enso_act_days)
tej_break_enso = tu.get_sel_tps_ds(
    tej_break_days, enso_break_days)

# Get sync days with condition
p_s_1_act_tej = tsa.get_cond_occ(
    tps=tps_sync, cond=tej_act_enso,
    counter=phase_vals)
print(p_s_1_act_tej)
p_s_1_break_tej = tsa.get_cond_occ(
    tps=tps_sync, cond=tej_break_enso,
    counter=phase_vals) * 0.8
num_samples = len(tu.get_sel_tps_ds(tps_sync.time, tej_act_enso.time)) + \
    len(tu.get_sel_tps_ds(tps_sync.time, tej_break_enso.time))
print(num_samples)

label_arr = ['P(MSDs|phase, active):\nProb. MSD per phase,\nBSISO active',
             'P(MSDs|phase, break):\nProb. MSD per phase, \nBSISO inactive',
             'Null model',
             ]
set_legend = True

im_bar = cplt.plot_xy(
    ax=im['ax'][2],
    y_title=y_title,
    title='Prob. of MSDs per BSISO phase',
    plot_type='bar',
    x_arr=phase_vals,
    xticks=phase_vals-1,
    y_arr=[p_s_1_act_tej,
           p_s_1_break_tej,
           null_model,
           ],
    set_legend=False,
    label_arr=label_arr,
    ylim=(0, 0.55),
    color_arr=['steelblue', 'firebrick', 'darkgray'],
    xlabel='BSISO Phase',
    ylabel='Likelihood')


order = [1, 2, 0]
cplt.set_legend(ax=im_bar['ax'],
                fig=im_bar['fig'],
                # order=order,
                loc='outside',
                ncol_legend=1,
                box_loc=(0.13, .45)
                )

savepath = plot_dir +\
    f"paper_plots/{region}_convection_cluster{group}.png"
cplt.save_fig(savepath)

# %%
# All probs combine all TEJ types and ENSO types
reload(tsa)
reload(cplt)
time_range = ['1981-01-01', '2019-12-31']
sregion = 'NI_SZ'
lag = 15

act_th = 1
break_th = .9
index_def = 'Kikuchi'
phase_vals = np.arange(1, 9, 1)

nrows = len(enso_types)
ncols = 2

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(12, 8),
                            wspace=0.2,
                            hspace=0.3,
                            )

# Synchronous and not synchronous days
tps_sync = tps_dict['tps_sync_jjas']
tps_sync = tu.get_time_range_data(tps_sync,
                                  time_range=time_range)

# BSISO Index Phase Active
bsiso_index = bs.get_bsiso_index(time_range=time_range,
                                 start_month='Jun',
                                 end_month='Sep',
                                 index_def=index_def)
times = bsiso_index.time
bsiso_phase = bsiso_index['BSISO-phase']
null_model_uniform = len(tps_sync) / len(times)
null_model = sut.count_occ([bsiso_phase.data], count_arr=phase_vals) / \
    len(bsiso_phase.time)*len(phase_vals) * null_model_uniform

ampl = bsiso_index['BSISO-ampl']

phase_sync = tu.get_sel_tps_ds(bsiso_phase, tps=tps_sync)

# Get all active/break days per phase
act_days_phase = xr.where(ampl >= act_th, bsiso_phase,
                          np.nan).dropna(dim='time')
break_days_phase = xr.where(
    ampl < break_th, bsiso_phase, np.nan).dropna(dim='time')

count = 0

for idx, (stype, tej_tps) in enumerate(tej_dict.items()):

    # include tej
    tej_act_days = tu.get_sel_tps_ds(ds=act_days_phase, tps=tej_tps)
    tej_break_days = tu.get_sel_tps_ds(ds=break_days_phase, tps=tej_tps)

    # include ENSO
    for e, (etype, enso_tps) in enumerate(enso_types.items()):
        print(f'{stype} {etype}')
        # include enso
        enso_act_days = tu.get_sel_tps_ds(ds=act_days_phase, tps=enso_tps)
        enso_break_days = tu.get_sel_tps_ds(ds=break_days_phase, tps=enso_tps)

        # get days that are ENSO + tej + active/break
        tej_act_enso = tu.get_sel_tps_ds(
            tej_act_days, enso_act_days)
        tej_break_enso = tu.get_sel_tps_ds(
            tej_break_days, enso_break_days)

        # Get sync days with condition
        p_s_1_act_tej = tsa.get_cond_occ(
            tps=tps_sync, cond=tej_act_enso,
            counter=phase_vals)
        print(p_s_1_act_tej)
        p_s_1_break_tej = tsa.get_cond_occ(
            tps=tps_sync, cond=tej_break_enso,
            counter=phase_vals)

        num_samples = len(tu.get_sel_tps_ds(tps_sync.time, tej_act_enso.time)) + \
            len(tu.get_sel_tps_ds(tps_sync.time, tej_break_enso.time))
        label_arr = ['P(MSDs| active, ENSO, TEJ):\nProb. of MSDs per phase,\ngiven BSISO active, ENSO, TEJ',
                     'P(MSDs| break, ENSO, TEJ):\nProbability of MSDs per phase,\ngiven BSISO inactive, ENSO, TEJ',
                     'Null model',
                     ]
        set_legend = True
        tej_types = ['enhanced TEJ', 'reduced TEJ']
        for pl in [True]:
            im_bars = cplt.plot_xy(
                ax=im['ax'][e*ncols+idx] if pl else None,
                plot_type='bar',
                x_arr=phase_vals,
                xticks=phase_vals-1,
                y_arr=[p_s_1_act_tej,
                       p_s_1_break_tej,
                       null_model,
                       ],
                set_legend=False,
                label_arr=label_arr,
                ylim=(0, 0.75),
                y_title=1.1,
                color_arr=['steelblue', 'firebrick', 'darkgray'],
                # {time_range[0]} - {time_range[1]}
                title=f'{tej_types[idx]}' if e == 0 else None,
                vertical_title=f'{etype}' if idx == 0 else None,
                xlabel='BSISO Phase' if e > 1 else None,
                ylabel='Likelihood' if idx == 0 else None,
                x_title_offset=-0.3,
            )

            # cplt.plot_hline(ax=im_bars['ax'],
            #                 y=0.26,
            #                 lw=2,
            #                 ls='dashed',
            #                 label='Null_model',
            #                 )
            if not pl or count == 5:
                cplt.set_legend(ax=im['ax'][-1] if pl else im_bars['ax'],
                                fig=im['fig'] if pl else None,
                                loc='outside',
                                ncol_legend=3 if pl else 2,
                                box_loc=(0.05, 0) if pl else (-0.12, -0.15)
                                )
                savepath = plot_dir +\
                    f"/bsiso/{sregion}_bsiso_phases_{index_def}_{comp_th}_enso_{etype}_tej_{stype}"
                if pl == False:
                    cplt.save_fig(fig=im_bars['fig'],
                                  savepath=savepath,
                                  extension='pdf')
            cplt.plt_text(ax=im_bars['ax'],
                          xpos=0.,
                          ypos=.7,
                          text=f'No. samples {num_samples}',
                          box=False)
        count += 1
savepath = plot_dir +\
    f"/paper_plots/bsiso_phases_tej_enso_ere{comp_th}_lags{num_lags}"
cplt.save_fig(fig=im['fig'],
              savepath=savepath,
              extension='pdf')


# %%
# BSISO QG forcing
reload(qgf)

lev_low = 400
lev_high = 800
lev = 600
qg_dict = qgf.get_qg_forcing(zlow=ds_z.ds['z'].sel(lev=lev_low),
                             zhigh=ds_z.ds['z'].sel(lev=lev_high),
                             zlev=ds_z.ds['z'].sel(lev=lev),
                             Tlev=ds_t.ds['t'].sel(lev=lev),
                             dp=(lev_high-lev_low), p=lev,
                             smooth=True,
                             n=10)

# The effect of the BSISO on the QG forcing (vertical velocity)
qg_forcing = qg_dict['QG forcing']
# qg_forcing = qg_dict['term1']
an_qg_forcing = tu.compute_anomalies(qg_forcing,
                                     group='month')
# %%
reload(cplt)
an_type = 'month'
nrows = 1
ncols = len(latent_cluster['keys']) + 2
q_oro = ds_orograph.ds['z'].quantile(q=0.95)
# himalayan_mask = xr.where(ds_orograph.ds['z'] < q_oro, 1, np.nan)

im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            lon_range=[40, 100],
                            lat_range=[10, 35],
                            wspace=0.1,
                            hspace=0.8,
                            dateline=False)

tps_arr = [latent_cluster[key] for key in latent_cluster['keys']] + \
    [tps_dict['active_ja']] + [tps_dict['break_years']]
groups = ['Cluster 0', 'Cluster 1', 'Sync years JA seasonal background',
          'Break years JA seasonal background']
for idx, this_tps in enumerate(tps_arr):
    print(f"Group {idx} has {len(this_tps)} time points")
    mean_data, sig_data = tu.get_mean_tps(an_qg_forcing,
                                          tps=this_tps,
                                          )
    cplt.plot_map(mean_data,  # *himalayan_mask,
                  ax=im['ax'][idx],
                  #   lon_range=[30, 100],
                  #   lat_range=[10, 40],
                  plot_type='contourf',
                  cmap='RdYlBu_r',
                  levels=14,
                  vmin=-1.5e-12, vmax=1.5e-12,
                  label=f'QG forcing Anomalies (wrt. month) {lev} hPa',
                  centercolor='white',
                  title=f'{groups[idx]} QG forcing',
                  #   significance_mask=himalayan_mask,
                  hatch_type='///',
                  inverse_mask=True,
                  )

    savepath = plot_dir +\
        f"clustering/an_qgforcing_{prefix_cluster}.png"
    cplt.save_fig(savepath)

# ################### Videos ###################
# %%
# OLR videos
reload(vid)
lon_range = [-30, 140]
lat_range = [-30, 60]
an_type = 'month'
var_type = f'an_{an_type}'
label = f'OLR Anomalies (wrt {an_type}) [W/m$^2$]'
vmin = -25
vmax = -vmin
group = 0
len_sub_sample = 10
tps = latent_cluster[group]
output_name = plot_dir +\
    f"animation/{prefix_cluster}_olr_{var_type}_group{group}_subsample"
format = 'mp4'
vid.create_video_map(
    ds_olr.ds[var_type],
    output_name=output_name,
    format=format,
    fr=1.5,
    end=15,
    tps=tps,
    plot_type='contourf',
    cmap='RdBu' if format == 'gif' else 'RdBu_r',
    levels=16,
    label=label,
    vmin=vmin, vmax=vmax,
    tick_step=2,
    centercolor='white',
    lat_range=lat_range,
    lon_range=lon_range,
    extend='both',
    plot_sig=False,
    delete_frames=False,
    folder_name='samples',
    title=True,
    plot_grid=False,
)
# %%
# Pr videos
lon_range = [-30, 140]
lat_range = [-30, 60]
an_type = 'month'
var_type = f'an_{an_type}'
label = f'Pr anomalies (wrt {an_type}) [mm/day]'
vmin = -5
vmax = -vmin
group = 0
tps = latent_cluster[group]
output_name = plot_dir +\
    f"animation/propagation_pr_{var_type}_group{group}"
vid.create_video_map(
    ds_pr.ds[var_type],
    output_name=output_name,
    format='mp4',
    fr=1.5,
    end=15,
    tps=tps,
    plot_type='contourf',
    cmap='RdBu',
    levels=16,
    label=label,
    vmin=vmin, vmax=vmax,
    tick_step=2,
    centercolor='white',
    lat_range=lat_range,
    lon_range=lon_range,
    extend='both',
    plot_sig=True,
)
# %%
# Wind Field videos
reload(vid)
lev = 200
lon_range = [-30, 160]
lat_range = [-20, 70]
an_type = 'month'
var_type_vs = f'v_an_{an_type}'
label_vs = f'V{lev}-winds (wrt {an_type}) [m/s]'
vmin = -5
vmax = -vmin

group = 0
tps = latent_cluster[group]
output_name = plot_dir +\
    f"animation/{prefix_cluster}_group{group}_{var_type_vs}_{lev}hPa"
vid.create_video_map(
    ds_wind.ds[var_type_vs].sel(lev=lev),
    output_name=output_name,
    format='mp4',
    fr=1.5,
    end=16,
    tps=tps,
    plot_type='contourf',
    cmap='PuOr',
    levels=16,
    label=label_vs,
    vmin=vmin, vmax=vmax,
    tick_step=2,
    centercolor='white',
    lat_range=lat_range,
    lon_range=lon_range,
    extend='both',
    plot_sig=True,
)


# %%
# Create a gridspec instance
reload(cplt)

pdo_index = pdo.get_pdo_index(
    time_range=['1979-01-01', '2021-01-01'],
    start_month='Jun',
    end_month='Sep')
pdo_index = tu.compute_timemean(pdo_index, timemean='year',
                                )
im = cplt.create_multi_plot(nrows=2, ncols=3,
                            hspace=0.4,
                            wspace=0.2,
                            full_length_row=True,
                            figsize=(22, 10),
                            proj_arr=['PlateCarree', 'PlateCarree',
                                      'PlateCarree', None],
                            end_idx=4,
                            central_longitude=180,
                            )

vmin_sst = -1
vmax_sst = -vmin_sst
an_type = 'month'
var_type = f'an_{an_type}'
label_sst = f'SST Anomalies (wrt {an_type}) [K]'

groups = ['Strong Propagation',
          'Weak Propagation',
          'Dry Years']

tps_arr = [
    latent_cluster[groups[0]],
    latent_cluster[groups[1]],
    tps_dict['break_years'],
]

for idx, sel_tps in enumerate(tps_arr):
    group = groups[idx]
    sel_tps = tu.get_time_range_data(sel_tps,
                                     time_range=['1987-01-01', '2020-12-31'])
    mean, mask = tu.get_mean_tps(ds_sst.ds[var_type], tps=sel_tps)

    im_sst = cplt.plot_map(mean,
                           ax=im['ax'][idx],
                           title=f'{group}',
                           cmap='RdBu_r',
                           plot_type='contourf',
                           levels=16,
                           centercolor='white',
                           vmin=vmin_sst, vmax=vmax_sst,
                           extend='both',
                           orientation='horizontal',
                           significance_mask=mask,
                           hatch_type='..',
                           label=label_sst,
                           lat_range=[-65, 65],
                           land_ocean=True,
                           tick_step=4,
                           )

for idx_cluster, key in enumerate(latent_cluster['keys']):
    this_tps = latent_cluster[key]
    yearly_tps = tu.count_time_points(time_points=this_tps, freq='Y',
                                      start_year=1979, end_year=2020)
    if idx_cluster == 0:
        yearly_tps = -1*yearly_tps
    yticks = np.arange(-10, 11, 5)
    ytick_labels = np.abs(yticks)

    im_pdo = cplt.plot_xy(
        ax=im['ax'][3],
        x_arr=tu.tps2str(yearly_tps.time, m=False, d=False),
        y_arr=[yearly_tps],
        label_arr=[key],
        color_arr=[cplt.colors[idx_cluster]],
        plot_type='bar',
        ylabel='No. samples/year',
        rot=90,
        figsize=(10, 3),
        # ylim=[-28, 28],
        yticks=yticks,
        yticklabels=ytick_labels,
        loc='upper left',
    )

cplt.plot_xy(x_arr=tu.tps2str(pdo_index.time, m=False, d=False),
             y_arr=[pdo_index],
             ax=im_pdo['ax'],
             color_arr=['magenta'],
             edge_color='magenta',
             label_arr=['PDO index'],
             ylabel='PDO Index',
             plot_type='xy',
             lw_arr=[2],
             set_twinx=True,
             set_axis=True,
             fill_bar=False,
             ylim=[-3, 3],
             rot=90,
             )
cplt.plot_hline(y=0, ax=im_pdo['ax'], color='black', lw=2, ls='solid')
# cplt.plot_vline(x='1986', ax=im_pdo['ax'], color='black', lw=2)

savepath = plot_dir +\
    f"paper_plots/sst_groups_pdo.png"
cplt.save_fig(savepath)
