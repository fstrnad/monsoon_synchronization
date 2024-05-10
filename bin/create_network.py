# %%
import geoutils.utils.file_utils as fut
import os
import climnet.network.clim_networkx as nx
import climnet.datasets.evs_dataset as cds
from importlib import reload
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import xarray as xr

# Run the null model

name = "mswep"
scale = "global"
grid_type = "fekete"
grid_step = 2.5

output_folder = "global_monsoon"  # for MSWEP maybe still in summer_monsoon

if os.getenv("HOME") == "/home/goswami/fstrnad80":
    output_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/outputs/"
    plot_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/plots/"
else:
    output_dir = "/home/strnad/data/climnet/outputs/"
    plot_dir = "/home/strnad/data/climnet/plots/"


# %%
reload(cds)
q_ee = 0.9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
min_evs = 20
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 3
dataset_file = output_dir + \
    f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"
# %%
reload(cds)
ds = cds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False,
)
time_steps = len(
    tu.get_month_range_data(
        ds.ds.time, start_month=start_month, end_month=end_month)
)

# %%
taumax = 10
n_pmts = 1000
weighted = True
networkfile = output_dir + f"{output_folder}/{name_prefix}_ES_net.npz"

E_matrix_folder = (
    f"{output_folder}/{name_prefix}_{q_ee}_{min_evs}/"
)
null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'


# %%
reload(nx)
reload(cds)
q_sig = 0.95
if fut.exist_file(dataset_file):
    ds = cds.EvsDataset(
        load_nc=dataset_file,
        rrevs=False,
    )
    null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'
else:
    raise ValueError(f'{dataset_file} does not exist!')
nx_path_file = output_dir + \
    f"{output_folder}/{name_prefix}_{q_sig}_ES_nx.gml.gz"
if not fut.exist_file(nx_path_file):

    Net = nx.Clim_NetworkX(dataset=ds,
                           taumax=taumax,
                           weighted=weighted,)
    gut.myprint(f"Use q = {q_sig}")
    Net.create(
        method='es',
        null_model_file=null_model_file,
        E_matrix_folder=E_matrix_folder,
        q_sig=q_sig
    )
    Net.save(nx_path_file)
    # else:
    #     Net = nx.Clim_NetworkX(dataset=ds, nx_path_file=nx_path_file)
    #     # Net.compute_link_lengths_edges()
    #     # cnx.compute_node_attrs('degree')
    #     # Net.compute_curvature(c_type='forman')
    #     # cnx.compute_curvature(c_type='ollivier')
    #     Net.save(savepath=nx_path_file)
