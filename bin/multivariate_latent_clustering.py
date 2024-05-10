# %%
import geoutils.indices.pdo_index as pdo
import geoutils.utils.xr_utils as xr_utils
import geoutils.utils.met_utils as mut
import geoutils.geodata.multilevel_pressure as mp
import geoutils.utils.spatial_utils as sput
import geoutils.tsa.pca.pca_utils as pca_utils
import geoutils.geodata.wind_dataset as wds
import geoutils.utils.statistic_utils as sut
import geoutils.tsa.time_clustering as tcl
from importlib import reload
import numpy as np

import geoutils.tsa.pca.eof as eof
import geoutils.tsa.pca.multivariate_pca as mvpca

import geoutils.plotting.plots as gpl
import geoutils.geodata.base_dataset as bds
import geoutils.utils.time_utils as tu
import geoutils.utils.file_utils as fut
import geoutils.utils.general_utils as gut

output_dir = "/home/strnad/data/"
data_dir = "/home/strnad/data/"
plot_dir = "/home/strnad/data/plots/summer_monsoon/"
output_folder = "summer_monsoon"


# %%
# Compare synchronization of BSISO1 and BSISO2 and ENSO
reload(tu)
reload(gut)
comp_th = 0.9

lag = 14
include_ts = False
if include_ts:
    include_region = 'SZ'
    savepath = plot_dir + \
        f"synchronization/synchronization_lag_{lag}_include_{include_region}_{comp_th}_ts_dict.npy"
else:
    num_lags = 10
    savepath = plot_dir + \
        f"synchronization/synchronization_lags_{num_lags}_{comp_th}_ts_dict.npy"

lagged_sync_dict = fut.load_np_dict(savepath)
sregion = 'NI_SZ'

# Synchronous and not synchronous days
tps_sync = lagged_sync_dict[sregion][lag]['tps_active']
tps_sync_ja = lagged_sync_dict[sregion][lag]['tps_sync_ja']
tps_sync_jjas = lagged_sync_dict[sregion][lag]['tps_sync_jjas']

lagged_index = lagged_sync_dict[sregion][lag]['sync_ts']
tr_jjas = tu.get_dates_of_time_range(['1979-01-01',
                                      '2021-01-01'],
                                     start_month='Jun',
                                     end_month='Sep')

tr_ja = tu.get_dates_of_time_range(['1979-01-01',
                                    '2021-01-01'],
                                   start_month='Jul',
                                   end_month='Aug')
tps_dict = {
    'lagged_index': lagged_index,
    'tps_sync': tu.remove_consecutive_tps(tps_sync,
                                          #   start=10,
                                          # steps=1,
                                          ),
    'tps_sync_ja': tu.remove_consecutive_tps(tps_sync_ja,
                                             #  start=10,
                                             #  steps=5
                                             ),
    'tps_sync_jjas': tu.remove_consecutive_tps(tps_sync_jjas,
                                               steps=1,
                                               ),
}

# %%
# OLR data
reload(bds)
grid_step = 1
dataset_file = data_dir + \
    f"climate_data/{grid_step}/era5_olr_{grid_step}_ds.nc"

ds_olr_eof = bds.BaseDataset(data_nc=dataset_file,
                             can=True,
                             month_range=['Jun', 'Sep'],
                             an_types=['month', 'JJAS'],
                             )

# %%
# Load MSE data
nc_files_q = []
nc_files_t = []
nc_files_z = []

plevels = [400, 700]

grid_step = 1
for plevel in plevels:
    dataset_file_q = data_dir + \
        f"/climate_data/{grid_step}/era5_q_{grid_step}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
    dataset_file_t = data_dir + \
        f"/climate_data/{grid_step}/era5_t_{grid_step}_{plevel}_ds.nc"
    nc_files_t.append(dataset_file_t)
    dataset_file_z = data_dir + \
        f"/climate_data/{grid_step}/era5_z_{grid_step}_{plevel}_ds.nc"
    nc_files_z.append(dataset_file_z)

# %%
reload(mp)
ds_q = mp.MultiPressureLevelDataset(data_nc=nc_files_q,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    month_range=['Jun', 'Sep'],
                                    plevels=plevels,
                                    time_range=ds_olr_eof.time_range,
                                    )
# %%
ds_t = mp.MultiPressureLevelDataset(data_nc=nc_files_t,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    month_range=['Jun', 'Sep'],
                                    plevels=plevels,
                                    metpy_unit='K',
                                    time_range=ds_olr_eof.time_range,
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
# Load wind fields
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []
levs = [200, 800]
# levs = np.arange(100, 1050, 100)
for lev in levs:
    dataset_file_u = data_dir + \
        f"/climate_data/{grid_step}/era5_u_{grid_step}_{lev}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/{grid_step}/era5_v_{grid_step}_{lev}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = data_dir + \
        f"/climate_data/{grid_step}/era5_w_{grid_step}_{lev}_ds.nc"
    nc_files_w.append(dataset_file_w)

reload(wds)
ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           #    data_nc_w=nc_files_w,
                           plevels=levs,
                           can=True,
                           an_types=['month', 'JJAS'],
                           month_range=['Jun', 'Sep'],
                           init_mask=False,
                           grid_step=1,
                           )
# %%
# Vertical winds
lev = 500
grid_step = 1
dataset_file_w = data_dir + \
    f"/climate_data/{grid_step}/era5_w_{grid_step}_{lev}_ds.nc"
ds_w_500 = bds.BaseDataset(data_nc=dataset_file_w,
                           can=True,
                           month_range=['Jun', 'Sep'],
                           an_types=['month', 'JJAS'],
                           grid_step=1,
                           )

# %%
reload(bds)
grid_step = 2.5
dataset_file = data_dir + \
    f"climate_data/{grid_step}/era5_ttr_{grid_step}_ds.nc"

ds_olr = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         month_range=['Jun', 'Sep'],
                         an_types=['month', 'JJAS'],
                         )

# %%
# SST
dataset_file = data_dir + \
    f"/climate_data/{grid_step}/era5_sst_{grid_step}_ds.nc"

ds_sst = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         an_types=['JJAS', 'month', 'dayofyear'],
                         month_range=['Jun', 'Sep'],
                         detrend=True,
                         )
# %%
# data
reload(gpl)
reload(eof)
# multiple variables
an_type = 'month'
var_dict = {
    'olr': dict(
        data=ds_olr_eof.ds[f'an_{an_type}'].rename('olr'),
    ),
    'u200': dict(
        data=ds_wind.ds[f'u_an_{an_type}'].sel(lev=200).rename('u200'),
    ),
    'v200': dict(
        data=ds_wind.ds[f'v_an_{an_type}'].sel(lev=200).rename('v200'),
    ),
    'u800': dict(
        data=ds_wind.ds[f'u_an_{an_type}'].sel(lev=800).rename('u800'),
    ),
    'v800': dict(
        data=ds_wind.ds[f'v_an_{an_type}'].sel(lev=800).rename('v800'),
    ),
    'rh400': dict(
        data=rh[f'rh_an_{an_type}'].sel(lev=400).rename('rh400'),
    ),
    'rh700': dict(
        data=rh[f'rh_an_{an_type}'].sel(lev=700).rename('rh700'),
    ),
    'w500': dict(
        data=ds_w_500.ds[f'an_{an_type}'].rename('w500'),
    ),
    'n_eofs': 1,  # n_eofs=2, n_cluster=2 is optimal choice
    'n_cluster': 2,
}
reload(xr_utils)
input_data = xr_utils.unify_datasets([
    # var_dict['olr']['data'],
    var_dict['u200']['data'],
    var_dict['v200']['data'],
    var_dict['rh400']['data'],
    var_dict['w500']['data'],
    # var_dict['rh700']['data'],
    # var_dict['u800']['data'],
    # var_dict['v800']['data'],
],)

lat_range_cut = [-0, 60]
lon_range_cut = [-20, 150]
input_data = sput.cut_map(
    input_data, lat_range=lat_range_cut, lon_range=lon_range_cut)

# %%
# Compute EOF
reload(eof)
reload(mvpca)
n_components = 10

sppca = mvpca.MultivariatePCA(input_data, n_components=n_components)
eofs = sppca.get_eofs()
pcs = sppca.get_principal_components()
vars = sppca.get_variables()
gut.myprint(f"Explained variance: {np.sum(sppca.explained_variance())}")

im = gpl.plot_xy(y_arr=sppca.explained_variance(),
                 xlabel='EOF', ylabel='Explained variance', ylog=False)

# EOF maps
reload(gpl)
reload(gut)
nrows = len(vars)
ncols = n_components
im = gpl.create_multi_plot(nrows=nrows,
                           ncols=ncols,
                           projection='PlateCarree',
                           wspace=0.2,
                           hspace=0.1,
                           #    plot_grid=False,
                           )
for i in range(sppca.n_components):
    comp = eofs.isel(eof=i)
    for j, var in enumerate(vars):
        im_eof = gpl.plot_map(comp[var],
                              ax=im['ax'][i + j * ncols],
                              #   vmin=-.03, vmax=.03,
                              cmap='Reds',
                              title=f'EOF {i}',
                              vertical_title=var if i == 0 else None,
                              label='EOF',
                              #   plot_grid=True
                              )
gpl.save_fig(f'{plot_dir}/latent_clustering/mveofs_{n_components}_{vars}.png')

# %%
# Latent clustering for multiple variables
reload(tcl)
method = 'kmeans'
steps = 15
n_eofs = 1
n_cluster = 2

sppca = mvpca.MultivariatePCA(input_data, n_components=20)
pcs = sppca.get_principal_components()
gut.myprint(f"Explained variance: {np.sum(sppca.explained_variance())}")
z_reduced = pca_utils.spatio_temporal_latent_volume(
    sppca, input_data,
    tps=tps_dict['tps_sync_jjas'],
    steps=steps,
    ts=lagged_index,
    num_eofs=n_eofs
)
cluster_names = ['Strong Propagation',
                 'Weak Propagation']
latent_cluster = tcl.apply_cluster_data(data=z_reduced.data,
                                        n_clusters=n_cluster,
                                        rm_ol=True,
                                        sc_th=0.05,
                                        random_state=42,
                                        method=method,
                                        standardize=True,
                                        plot_statistics=True,
                                        return_model=False,
                                        objects=z_reduced.time,
                                        cluster_names=cluster_names,
                                        )
variables = sppca.get_variables()
name_prefix = f"latent_mvpca_{variables}_eof{n_eofs}_n{n_cluster}_{method}_lags_{num_lags}_{comp_th}"

savepath = plot_dir + \
    f"latent_clustering/{name_prefix}.npy"
fut.save_np_dict(latent_cluster, sp=savepath)


# %%
# Plot the propagation of the synchronization
# OLR + vertical velocity plots
reload(gpl)
reload(tu)
lon_range = [-30, 140]
lat_range = [-20, 60]

an_type = 'month'
var_type_olr = f'an_{an_type}'

label_olr = f'Anomalies OLR (wrt {an_type}) [W/m²]'
vmax_olr = 2.e1
vmin_olr = -vmax_olr

step = 3
steps = np.arange(0, 16, step)
nrows = n_cluster
ncols = len(steps)

im = gpl.create_multi_plot(nrows=nrows,
                           ncols=ncols,
                           orientation='horizontal',
                           hspace=0.8,
                           wspace=0.,
                           projection='PlateCarree',
                           lat_range=lat_range,
                           lon_range=lon_range,
                           )
for idx, group in enumerate(list(latent_cluster['keys'])):
    tps = latent_cluster[group]
    print(f"Group {group} has {len(tps)} time points")
    for d, start in enumerate(steps):

        sel_tps = tu.add_time_step_tps(tps, time_step=start)

        mean_data, sig_data = tu.get_mean_tps(ds_olr.ds[var_type_olr],
                                              tps=sel_tps,
                                              corr_type=None)
        gpl.plot_map(mean_data*sig_data,
                     ax=im['ax'][ncols*idx + d],
                     title=f'Day {start}',
                     vertical_title=f'{group}' if d == 0 else None,
                     plot_type='contourf',
                     cmap='RdBu_r',
                     levels=12,
                     #  label=label_olr,
                     vmin=vmin_olr, vmax=vmax_olr,
                     tick_step=2,
                     centercolor='white',
                     #   vertical_title='OLR' if d == 0 else None,
                     extend='both',
                     )

savepath = plot_dir +\
    f"latent_clustering/{method}_olr_propagation_pattern_mvpca_{vars}.png"
gpl.save_fig(savepath)
# %%
# Distribution over the years
# Analyse time points per year
reload(tu)
reload(gpl)

for idx, key in enumerate(latent_cluster['keys']):
    this_tps = latent_cluster[key]
    yearly_tps = tu.count_time_points(time_points=this_tps, freq='Y',
                                      start_year=1979, end_year=2020)
    if idx == 0:
        yearly_tps = -1*yearly_tps
    yticks = np.arange(-20, 21, 10)
    ytick_labels = np.abs(yticks)

    im = gpl.plot_xy(
        ax=im['ax'] if idx != 0 else None,
        x_arr=tu.tps2str(yearly_tps.time, m=False, d=False),
        y_arr=[yearly_tps],
        label_arr=[key],
        color_arr=['red'if idx == 0 else 'blue'],
        plot_type='bar',
        ylabel='No. samples/year',
        rot=90,
        figsize=(10, 3),
        ylim=[-14, 14],
        # yticks=yticks,
        # yticklabels=ytick_labels,
        loc='upper left',
    )


pdo_index = pdo.get_pdo_index()
pdo_index = tu.get_time_range_data(pdo_index,
                                   ['1979-01-01', '2021-01-01'],
                                   start_month='Jun',
                                   end_month='Sep')
pdo_index = tu.compute_timemean(pdo_index, timemean='year',
                                )

gpl.plot_hline(y=0, ax=im['ax'], color='black', lw=2)
gpl.plot_xy(x_arr=tu.tps2str(pdo_index.time, m=False, d=False),
            y_arr=[pdo_index],
            ax=im['ax'],
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

savepath = plot_dir +\
    f"latent_clustering/yearly_occurence_groups_pdo.png"
gpl.save_fig(savepath)


# %%
# SST state clustering
reload(gpl)
reload(sut)
ncols = len(latent_cluster['keys'])
nrows = 1
im = gpl.create_multi_plot(nrows=nrows,
                           ncols=ncols,
                           orientation='horizontal',
                           hspace=0.7,
                           wspace=0.2,
                           projection='PlateCarree',
                           lat_range=[-50, 70],
                           lon_range=[30, -60],
                           dateline=True,
                           )
vmin_sst = -1
vmax_sst = -vmin_sst
an_type = 'month'
var_type = f'an_{an_type}'
label_sst = f'SST Anomalies (wrt {an_type}) [°C]'
for idx, (group) in enumerate(latent_cluster['keys']):

    sel_tps = latent_cluster[group]
    mean, mask = tu.get_mean_tps(ds_sst.ds[var_type], tps=sel_tps)

    im_sst = gpl.plot_map(mean,
                          ax=im['ax'][idx],
                          title=f'Group {group}',
                          cmap='RdBu_r',
                          plot_type='contourf',
                          levels=14,
                          centercolor='white',
                          vmin=vmin_sst, vmax=vmax_sst,
                          extend='both',
                          orientation='horizontal',
                          significance_mask=mask,
                          hatch_type='..',
                          label=label_sst,
                          )

savepath = plot_dir +\
    f"latent_clustering/sst_groups_lags{num_lags}_k{n_cluster}.png"
gpl.save_fig(savepath)


# %%
# Plot the propagation of the synchronization
# U winds
reload(gpl)
reload(tu)
lon_range = [-30, 140]
lat_range = [-20, 60]

an_type = 'month'
var_type_olr = f'u_an_{an_type}'

label_olr = f'Anomalies U (wrt {an_type}) [m/s]'
vmax_olr = 5
vmin_olr = -vmax_olr

step = 3
steps = np.arange(0, 16, step)
nrows = n_cluster
ncols = len(steps)

im = gpl.create_multi_plot(nrows=nrows,
                           ncols=ncols,
                           orientation='horizontal',
                           hspace=0.8,
                           wspace=0.,
                           projection='PlateCarree',
                           lat_range=lat_range,
                           lon_range=lon_range,
                           )
for idx, group in enumerate(list(latent_cluster['keys'])):
    tps = latent_cluster[group]
    print(f"Group {group} has {len(tps)} time points")
    for d, start in enumerate(steps):

        sel_tps = tu.add_time_step_tps(tps, time_step=start)

        mean_data, sig_data = tu.get_mean_tps(ds_wind.ds[var_type_olr].sel(lev=200),
                                              tps=sel_tps,
                                              corr_type=None)
        gpl.plot_map(mean_data*sig_data,
                     ax=im['ax'][ncols*idx + d],
                     title=f'Day {start}',
                     vertical_title=f'Group {group}' if d == 0 else None,
                     plot_type='contourf',
                     cmap='RdBu_r',
                     levels=12,
                     #  label=label_olr,
                     vmin=vmin_olr, vmax=vmax_olr,
                     tick_step=2,
                     centercolor='white',
                     #   vertical_title='OLR' if d == 0 else None,
                     extend='both',
                     )

savepath = plot_dir +\
    f"latent_clustering/u_propagation_pattern_mvpca_{vars}.png"
gpl.save_fig(savepath)


# %%
# Plot the propagation of the synchronization
# v winds
reload(gpl)
reload(tu)
lon_range = [-30, 180]
lat_range = [-20, 60]

an_type = 'month'
var_type_olr = f'v_an_{an_type}'

label_olr = f'Anomalies V (wrt {an_type}) [m/s]'
vmax_olr = 5
vmin_olr = -vmax_olr

step = 3
steps = np.arange(0, 16, step)
nrows = n_cluster
ncols = len(steps)

im = gpl.create_multi_plot(nrows=nrows,
                           ncols=ncols,
                           orientation='horizontal',
                           hspace=0.8,
                           wspace=0.,
                           projection='PlateCarree',
                           lat_range=lat_range,
                           lon_range=lon_range,
                           )
for idx, group in enumerate(list(latent_cluster['keys'])):
    tps = latent_cluster[group]
    print(f"Group {group} has {len(tps)} time points")
    for d, start in enumerate(steps):

        sel_tps = tu.add_time_step_tps(tps, time_step=start)

        mean_data, sig_data = tu.get_mean_tps(ds_wind.ds[var_type_olr].sel(lev=200),
                                              tps=sel_tps,
                                              corr_type=None)
        gpl.plot_map(mean_data*sig_data,
                     ax=im['ax'][ncols*idx + d],
                     title=f'Day {start}',
                     vertical_title=f'Group {group}' if d == 0 else None,
                     plot_type='contourf',
                     cmap='RdBu_r',
                     levels=12,
                     #  label=label_olr,
                     vmin=vmin_olr, vmax=vmax_olr,
                     tick_step=2,
                     centercolor='white',
                     #   vertical_title='OLR' if d == 0 else None,
                     extend='both',
                     )

savepath = plot_dir +\
    f"latent_clustering/v_propagation_pattern_mvpca_{vars}.png"
gpl.save_fig(savepath)


# %%
# Distribution over the years
# Analyse time points per year
reload(tu)
reload(gpl)

yearly_tps_groups = []
for key in latent_cluster['keys']:
    this_tps = latent_cluster[key]
    yearly_tps = tu.count_time_points(time_points=this_tps, freq='Y',
                                      start_year=1979, end_year=2020)
    yearly_tps_groups.append(yearly_tps)
x_arr = tu.tps2str(yearly_tps_groups[0].time, m=False, d=False)
gpl.plot_xy(
    # x_arr=yearly_tps_groups[0].time,
    #    yearly_tps_groups[1].time,
    #    pdo_index.time
    #    ,
    x_arr=x_arr,
    y_arr=yearly_tps_groups,  # + [pdo_index],
    label_arr=list(latent_cluster.keys()) + ['PDO'],
    plot_type='bar',
    ylabel='No. samples/year',
    rot=90,
    # stdize=True,
    figsize=(10, 3),
)
savepath = plot_dir +\
    f"latent_clustering/yearly_occurence_groups_{vars}.png"
gpl.save_fig(savepath)
# %%
