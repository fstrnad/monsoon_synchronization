#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as cplt
from importlib import reload
import climnet.datasets.evs_dataset as cds

import os

reload(cds)

name = 'mswep'
grid_type = 'fekete'
grid_step = 2.5

scale = 'global'

vname = 'pr'

start_month = 'Jun'
end_month = 'Sep'

output_folder = 'global_monsoon'

if os.getenv("HOME") == '/home/goswami/fstrnad80':
    output_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/outputs/"
    plot_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/plots/"
    data_dir = "/mnt/qb/goswami/processed_data/climate_data/"
else:
    output_dir = '/home/strnad/data/climnet/outputs/'
    plot_dir = '/home/strnad/data/climnet/plots/'
    data_dir = "/home/strnad/data/climate_data/"


if name == 'mswep':
    fname = data_dir + 'mswep_pr_1_1979_2021_ds.nc'
elif name == 'trmm':
    fname = data_dir + 'trmm_pr_1_1998_2020_ds.nc'

th_eev = 15

q_ee = 0.9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
min_evs = 20
if start_month != 'Jan' or end_month != 'Dec':
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 3

can = False

dataset_file = output_dir + \
    f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"

sp_large_ds = f"{data_dir}/{name}_{vname}_{grid_step}_ds.nc"

# %%
reload(cds)
sp_grid = f'{grid_type}_{grid_step}.npy'
ds = cds.EvsDataset(fname,
                    # time_range=time_range,
                    month_range=[start_month, end_month],
                    grid_step=grid_step,
                    grid_type=grid_type,
                    # large_ds=True,
                    # sp_large_ds=sp_large_ds,
                    sp_grid=sp_grid,
                    q=q_ee,
                    th_eev=th_eev,
                    min_evs=min_evs,
                    )
# %%
ds.save(dataset_file)
