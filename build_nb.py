import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "earth2studio-venv", "display_name": "Python (earth2studio venv)", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# AIWP-driven StormScope nowcasting over Wyoming

This notebook chains a global **AI weather prediction (AIWP)** model into **StormScope**,
NVIDIA's diffusion-based satellite/radar nowcaster, and looks at the result over **Wyoming**.

Pipeline:

1. **AIWP stage** – FourCastNet 3 (FCN3) initialised from GFS analysis, wrapped with the
   `InterpModAFNO` temporal interpolator so we get hourly output, and with a surface-pressure
   diagnostic so the interpolator has all 73 channels it needs. We keep only `z500`.
2. **StormScope GOES** (`6km_1hr` variant) – forecasts eight GOES ABI channels on the 6 km
   HRRR grid. This variant is *conditioned on synoptic-scale `z500`*, which is exactly where
   the AIWP forecast plugs in (replacing the GFS forecast normally used).
3. **StormScope MRMS** (`6km_1hr` variant) – forecasts composite reflectivity, conditioned on
   the GOES imagery that the GOES model produced in the same rollout step.
4. **Wyoming analysis** – zoomed maps, a small verification against observed MRMS, and a NetCDF
   export of the Wyoming cut-out.

Notes

* StormScope is a CONUS model on a fixed grid; "over Wyoming" therefore means we run the full
  CONUS domain and analyse/crop the Wyoming sub-region (41–45°N, 104–111°W).
* Only the `6km_1hr` StormScope variant takes external `z500` conditioning. The newer
  `3km_10min` nowcaster is "pure observation" and cannot be coupled to an AIWP model, so the
  legacy variant is used on purpose (you will see a deprecation warning when it loads).
* Runtime: FCN3 + interpolation is fast; each StormScope step is a 100-step diffusion sample for
  two models, roughly a minute per hour of forecast on a single GB200/H100.

Environment used to build this notebook: `venvs/earth2studio` (earth2studio fork at
`forks/earth2studio`, NATTEN with libnatten, `earth2studio[fcn3,stormscope]` extras, cartopy).
""")

md("## Setup")
code(r"""import os
from collections import OrderedDict
from datetime import datetime

os.makedirs("outputs", exist_ok=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.auto import tqdm

from earth2studio.data import GFS, GOES, MRMS, InferenceOutputSource, fetch_data
from earth2studio.io import XarrayBackend
from earth2studio.models.dx import DerivedSurfacePressure
from earth2studio.models.px import FCN3, DiagnosticWrapper, InterpModAFNO
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES, StormScopeMRMS
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")""")

md("""## Case configuration

A summer afternoon on the High Plains. FCN3 runs from the 12 UTC GFS analysis; StormScope is
initialised six hours later at 18 UTC (noon local) from GOES-16 and MRMS observations and run
for six hourly steps into the evening convective period.
""")
code(r"""START_TIME_AIWP = datetime(2024, 7, 17, 12)        # FCN3 init: must be on a 6 h boundary
START_TIME_STORMSCOPE = datetime(2024, 7, 17, 18)  # StormScope init: hourly, >= AIWP init
N_STEPS = 6                                        # hourly StormScope steps
SEED = 1234

# Wyoming bounding box (degrees; lon in [-180, 180])
WY = dict(lat_min=41.0, lat_max=45.0, lon_min=-111.05, lon_max=-104.05)

# Major Wyoming cities/towns to mark on the maps: name -> (lat, lon)
WY_CITIES = {
    "Cheyenne": (41.140, -104.820),
    "Laramie": (41.311, -105.591),
    "Casper": (42.867, -106.313),
    "Gillette": (44.291, -105.502),
    "Rock Springs": (41.587, -109.203),
    "Sheridan": (44.797, -106.956),
    "Jackson": (43.480, -110.762),
    "Evanston": (41.268, -110.963),
    "Riverton": (43.025, -108.380),
    "Cody": (44.526, -109.056),
}

time_aiwp = to_time_array([START_TIME_AIWP])
time_ss = to_time_array([START_TIME_STORMSCOPE])
lead_times = np.array([np.timedelta64(i, "h") for i in range(N_STEPS + 1)])
model_gap_h = int((time_ss[0] - time_aiwp[0]) / np.timedelta64(1, "h"))
assert model_gap_h >= 0 and model_gap_h % 1 == 0
print(f"AIWP init {START_TIME_AIWP}  ->  StormScope init {START_TIME_STORMSCOPE} (gap {model_gap_h} h), {N_STEPS} steps")""")

md("""## 1. AIWP stage: FCN3 with hourly interpolation

`InterpModAFNO` needs surface pressure, which FCN3 does not predict, so FCN3 is wrapped with a
`DerivedSurfacePressure` diagnostic (hydrostatic interpolation from geopotential and temperature)
before being handed to the interpolator. This mirrors the FCN3+StormScope serving workflow in
earth2studio.
""")
code(r"""package_fcn3 = FCN3.load_default_package()
fcn3 = FCN3.load_model(package_fcn3)

with xr.open_dataset(package_fcn3.resolve("orography.nc")) as ds:
    z_surface = torch.as_tensor(ds["Z"][0].values)
z_surf_coords = OrderedDict({d: fcn3.input_coords()[d] for d in ["lat", "lon"]})
sp_model = DerivedSurfacePressure(
    p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    surface_geopotential=z_surface,
    surface_geopotential_coords=z_surf_coords,
)
fcn3_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)

aiwp = InterpModAFNO.from_pretrained()
aiwp.px_model = fcn3_sp
aiwp = aiwp.to(device).eval()

aiwp_out_coords = aiwp.output_coords(aiwp.input_coords())
lat_aiwp, lon_aiwp = aiwp_out_coords["lat"], aiwp_out_coords["lon"]
print("AIWP output grid:", lat_aiwp.shape, lon_aiwp.shape, "| variables:", len(aiwp_out_coords["variable"]))""")

code(r"""x_aiwp, coords_aiwp = fetch_data(
    GFS(),
    time=time_aiwp,
    variable=aiwp.input_coords()["variable"],
    device=device,
)
print(x_aiwp.shape, {k: v.shape for k, v in coords_aiwp.items()})""")

md("""Run the interpolated FCN3 forecast far enough to cover the StormScope window and store the
hourly `z500` fields in an in-memory `XarrayBackend`. Time coordinates are re-based so that
`time` is the StormScope initialisation and `lead_time` runs 0…N h; this is what StormScope will
query for conditioning.
""")
code(r"""z500_var = np.array(["z500"])
io_aiwp = XarrayBackend()
io_aiwp.add_array(
    OrderedDict(time=time_ss, lead_time=lead_times, lat=lat_aiwp, lon=lon_aiwp), z500_var
)

fcn3.set_rng(seed=SEED)
n_total = model_gap_h + N_STEPS
for step, (x, cx) in tqdm(
    enumerate(aiwp.create_iterator(x_aiwp.clone(), coords_aiwp.copy())),
    total=n_total + 1, desc="FCN3 + InterpModAFNO (hourly)",
):
    if step >= model_gap_h:
        x, cx = map_coords(x, cx, OrderedDict({"variable": z500_var}))
        cx = cx.copy()
        cx["time"] = time_ss
        cx["lead_time"] = cx["lead_time"] - np.timedelta64(model_gap_h, "h")
        io_aiwp.write(*split_coords(x, cx))
    if step == n_total:
        break

ds_z500 = io_aiwp.root
ds_z500""")

md("## 2. StormScope models conditioned on the AIWP forecast")
code(r"""# Expose the stored AIWP z500 as an earth2studio data source keyed by valid time
z500_source = InferenceOutputSource(ds_z500)

package_ss = StormScopeBase.load_default_package()
print(StormScopeGOES.list_available_models(package_ss))

# GOES model: conditioned on z500 -> conditioning source is the AIWP forecast
model_goes = StormScopeGOES.load_model(
    package=package_ss, model_name="6km_1hr", conditioning_data_source=z500_source, amp=True
).to(device).eval()

# MRMS model: conditioned on GOES, which we pass explicitly during the coupled rollout
model_mrms = StormScopeMRMS.load_model(
    package=package_ss, model_name="6km_1hr", conditioning_data_source=None, amp=True
).to(device).eval()

print("GOES state:", model_goes.variables, "| conditioning:", model_goes.conditioning_variables)
print("MRMS state:", model_mrms.variables, "| conditioning:", model_mrms.conditioning_variables)
print("Model grid:", tuple(model_goes.latitudes.shape))""")

md("""### Initial conditions and regridding

StormScope works on a (downsampled) HRRR Lambert-conformal grid. Each model owns nearest-neighbour
interpolators: one for its own state (GOES or MRMS native grid → model grid) and one for its
conditioning (the AIWP 0.25° lat/lon grid for GOES; the GOES grid for MRMS).
""")
code(r"""goes_sat, scan_mode = "goes16", "C"   # GOES-16 was GOES-East in 2024 (GOES-19 from April 2025)
goes = GOES(satellite=goes_sat, scan_mode=scan_mode)
goes_lat, goes_lon = GOES.grid(satellite=goes_sat, scan_mode=scan_mode)

model_goes.build_input_interpolator(goes_lat, goes_lon)
model_goes.build_conditioning_interpolator(lat_aiwp, lon_aiwp)

in_coords_goes = model_goes.input_coords()
x_goes, coords_goes = fetch_data(
    goes, time=time_ss, variable=in_coords_goes["variable"],
    lead_time=in_coords_goes["lead_time"], device=device,
)

mrms = MRMS()
in_coords_mrms = model_mrms.input_coords()
x_mrms, coords_mrms = fetch_data(
    mrms, time=time_ss, variable=in_coords_mrms["variable"],
    lead_time=in_coords_mrms["lead_time"], device=device,
)
model_mrms.build_input_interpolator(coords_mrms["lat"], coords_mrms["lon"])
model_mrms.build_conditioning_interpolator(goes_lat, goes_lon)

# Add the batch dimension [B, T, L, C, H, W]; B>1 would give an ensemble
for name in ("goes", "mrms"):
    x, c = (x_goes, coords_goes) if name == "goes" else (x_mrms, coords_mrms)
    if x.dim() == 5:
        x = x.unsqueeze(0)
        c["batch"] = np.arange(1)
        c.move_to_end("batch", last=False)
    x = x.to(torch.float32)
    if name == "goes":
        x_goes, coords_goes = x, c
    else:
        x_mrms, coords_mrms = x, c
print("GOES IC", tuple(x_goes.shape), "| MRMS IC", tuple(x_mrms.shape))""")

md("""### Coupled rollout

Each hour: the GOES model advances the satellite state (it fetches the AIWP `z500` for the
current valid time internally through its conditioning source), then the MRMS model advances the
radar state conditioned on the *current* GOES state. Predictions are collected on the model grid.
""")
code(r"""lat_m = model_goes.latitudes.cpu().numpy()
lon_m = model_goes.longitudes.cpu().numpy()
lon_m180 = np.where(lon_m > 180, lon_m - 360, lon_m)
i_ir = list(model_goes.variables).index("abi13c")
i_refc = list(model_mrms.variables).index("refc")

torch.manual_seed(SEED)
y, yc = x_goes, coords_goes
ym, ymc = x_mrms, coords_mrms

goes_frames, mrms_frames, valid_times = [], [], []
for step in tqdm(range(N_STEPS), desc="StormScope coupled rollout"):
    y_pred, y_pred_c = model_goes(y, yc)
    ym_pred, ym_pred_c = model_mrms.call_with_conditioning(ym, ymc, conditioning=y, conditioning_coords=yc)

    goes_frames.append(torch.where(model_goes.valid_mask, y_pred, torch.nan)[0, 0, 0].cpu())
    mrms_frames.append(torch.where(model_mrms.valid_mask, ym_pred, torch.nan)[0, 0, 0].cpu())
    valid_times.append(y_pred_c["time"][0] + y_pred_c["lead_time"][0])

    y, yc = model_goes.next_input(y_pred, y_pred_c, y, yc)
    ym, ymc = model_mrms.next_input(ym_pred, ym_pred_c, ym, ymc)

goes_fc = torch.stack(goes_frames).numpy()   # [N_STEPS, C_goes, H, W]
mrms_fc = torch.stack(mrms_frames).numpy()   # [N_STEPS, C_mrms, H, W]
valid_times = np.array(valid_times)
print(goes_fc.shape, mrms_fc.shape, valid_times)""")

md("""## 3. Wyoming analysis

### CONUS context with the Wyoming box
""")
code(r"""proj_hrrr = ccrs.LambertConformal(
    central_longitude=262.5, central_latitude=38.5, standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)
WY_EXTENT = [WY["lon_min"] - 0.5, WY["lon_max"] + 0.5, WY["lat_min"] - 0.5, WY["lat_max"] + 0.5]

def add_borders(ax):
    ax.coastlines(color="black", linewidth=1.0)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.8)
    ax.coastlines(color="white", linewidth=0.35)
    ax.add_feature(cfeature.STATES, edgecolor="white", linewidth=0.3)

def draw_wy_box(ax, **kw):
    lons = [WY["lon_min"], WY["lon_max"], WY["lon_max"], WY["lon_min"], WY["lon_min"]]
    lats = [WY["lat_min"], WY["lat_min"], WY["lat_max"], WY["lat_max"], WY["lat_min"]]
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kw)

def add_cities(ax, color="white", edge="black", fontsize=7, label=True):
    # Mark the Wyoming cities with a dot and (optionally) a name label
    for name, (lat, lon) in WY_CITIES.items():
        ax.plot(lon, lat, marker="o", markersize=4, markerfacecolor=color, markeredgecolor=edge,
                markeredgewidth=0.8, linestyle="none", transform=ccrs.PlateCarree(), zorder=10)
        if label:
            ax.annotate(name, xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(4, 3), textcoords="offset points", fontsize=fontsize, color=color,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground=edge)], zorder=11)

def plot_goes_mrms(ax, ir, refc, title):
    im = ax.pcolormesh(lon_m, lat_m, ir, transform=ccrs.PlateCarree(), cmap="gray_r", shading="auto", vmin=200, vmax=300)
    refc_masked = np.where(refc <= 5, np.nan, refc)
    im2 = ax.pcolormesh(lon_m, lat_m, refc_masked, transform=ccrs.PlateCarree(), cmap="magma", shading="auto", vmin=5, vmax=60)
    add_borders(ax)
    ax.set_title(title, fontsize=10)
    return im, im2

k = N_STEPS - 1
fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=proj_hrrr)
im, im2 = plot_goes_mrms(ax, goes_fc[k, i_ir], mrms_fc[k, i_refc],
                         f"StormScope (AIWP-conditioned) valid {str(valid_times[k])[:16]} UTC, lead +{k+1} h")
draw_wy_box(ax, color="#2a78d6", linewidth=2)
add_cities(ax, label=False)
plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.45, label="GOES ABI-13 clean IR brightness temperature [K]")
plt.colorbar(im2, ax=ax, orientation="horizontal", pad=0.08, shrink=0.45, label="MRMS composite reflectivity [dBZ]")
plt.tight_layout(); plt.savefig("outputs/conus_final_step.png", dpi=150); plt.show()""")

md("### AIWP conditioning: FCN3 `z500` over the western US")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), subplot_kw=dict(projection=ccrs.PlateCarree()))
for ax, li in zip(axes, [0, N_STEPS]):
    z = ds_z500["z500"].isel(time=0, lead_time=li) / 9.80665  # geopotential -> geopotential height [m]
    z = z.sel(lat=slice(55, 25))
    z = z.assign_coords(lon=(((z.lon + 180) % 360) - 180)).sortby("lon").sel(lon=slice(-125, -90))
    cf = ax.contourf(z.lon, z.lat, z, levels=np.arange(5600, 6001, 20), cmap="Blues_r", extend="both")
    ax.contour(z.lon, z.lat, z, levels=np.arange(5600, 6001, 60), colors="black", linewidths=0.5)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
    ax.coastlines(linewidth=0.7)
    draw_wy_box(ax, color="#eb6834", linewidth=2)
    vt = str(np.datetime64(time_ss[0]) + lead_times[li])[:16]
    ax.set_title(f"valid {vt} UTC (+{model_gap_h + li} h from GFS init)", fontsize=10)
fig.suptitle("FCN3 (AIWP) 500 hPa geopotential height used to condition StormScope", y=0.93)
fig.colorbar(cf, ax=axes, orientation="horizontal", shrink=0.5, pad=0.06, label="500 hPa geopotential height [m]")
plt.savefig("outputs/aiwp_z500.png", dpi=150); plt.show()""")

md("""### Hourly Wyoming zoom: forecast reflectivity on forecast IR

Major towns (Cheyenne, Laramie, Casper, Gillette, Rock Springs, Sheridan, Jackson, Evanston,
Riverton, Cody) are marked for orientation.""")
code(r"""ncol = 3
nrow = int(np.ceil(N_STEPS / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.9 * nrow), subplot_kw=dict(projection=proj_hrrr))
for k, ax in enumerate(axes.ravel()):
    if k >= N_STEPS:
        ax.set_visible(False); continue
    im, im2 = plot_goes_mrms(ax, goes_fc[k, i_ir], mrms_fc[k, i_refc], f"+{k+1} h  valid {str(valid_times[k])[11:16]} UTC")
    ax.set_extent(WY_EXTENT, crs=ccrs.PlateCarree())
    add_cities(ax, fontsize=6)
fig.suptitle(f"StormScope forecast over Wyoming, init {START_TIME_STORMSCOPE:%Y-%m-%d %H} UTC (z500 from FCN3)", y=1.0)
fig.colorbar(im2, ax=axes, orientation="horizontal", shrink=0.4, pad=0.03, label="Forecast MRMS composite reflectivity [dBZ]")
plt.savefig("outputs/wyoming_hourly.png", dpi=150); plt.show()""")

md("""### Verification against observed MRMS over Wyoming

Fetch the observed composite reflectivity at each valid time, regrid it with the MRMS model's own
nearest-neighbour interpolator so both fields live on the same grid, and compare inside the
Wyoming box. This is a single deterministic sample, so treat the numbers as a sanity check, not a
skill score.
""")
code(r"""x_obs, c_obs = fetch_data(mrms, time=time_ss, variable=np.array(["refc"]), lead_time=lead_times[1:], device=device)
x_obs = model_mrms.input_interp(x_obs.to(torch.float32))          # -> [T, L, C, H, W] on the model grid
mrms_obs = torch.where(model_mrms.valid_mask, x_obs, torch.nan)[0, :, 0].cpu().numpy()  # [N_STEPS, H, W]

wy_mask = (lat_m >= WY["lat_min"]) & (lat_m <= WY["lat_max"]) & (lon_m180 >= WY["lon_min"]) & (lon_m180 <= WY["lon_max"])
print(f"Wyoming box: {wy_mask.sum()} model grid points")

def box_stats(field):
    v = field[wy_mask]
    v = v[np.isfinite(v)]
    return dict(max_dBZ=float(np.nanmax(v)), frac_ge35=float(np.mean(v >= 35)), frac_ge20=float(np.mean(v >= 20)))

rows = []
for k in range(N_STEPS):
    f, o = box_stats(mrms_fc[k, i_refc]), box_stats(mrms_obs[k])
    rows.append(dict(lead_h=k + 1, valid=str(valid_times[k])[:16],
                     fc_max=f["max_dBZ"], obs_max=o["max_dBZ"],
                     fc_frac_ge35=f["frac_ge35"], obs_frac_ge35=o["frac_ge35"],
                     fc_frac_ge20=f["frac_ge20"], obs_frac_ge20=o["frac_ge20"]))
verif = pd.DataFrame(rows).set_index("lead_h")
verif.round(3)""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
h = verif.index.values
for ax, col, ylabel in [(axes[0], "frac_ge20", "Fraction of Wyoming with refc ≥ 20 dBZ"),
                        (axes[1], "max_dBZ", "Max composite reflectivity in Wyoming [dBZ]")]:
    fc_col, obs_col = (f"fc_{col}", f"obs_{col}") if col != "max_dBZ" else ("fc_max", "obs_max")
    ax.plot(h, verif[fc_col], color="#2a78d6", linewidth=2, marker="o", markersize=5, label="StormScope forecast")
    ax.plot(h, verif[obs_col], color="#eb6834", linewidth=2, marker="o", markersize=5, label="MRMS observed")
    ax.set_xlabel("Lead time [h]"); ax.set_ylabel(ylabel)
    ax.set_xticks(h); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle("Wyoming box: forecast vs observed reflectivity")
plt.tight_layout(); plt.savefig("outputs/wyoming_verification.png", dpi=150); plt.show()""")

code(r"""# Side-by-side forecast vs observed reflectivity over Wyoming at the final step
k = N_STEPS - 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), subplot_kw=dict(projection=proj_hrrr))
for ax, field, name in [(axes[0], mrms_fc[k, i_refc], "StormScope forecast"), (axes[1], mrms_obs[k], "MRMS observed")]:
    im2 = ax.pcolormesh(lon_m, lat_m, np.where(field <= 5, np.nan, field), transform=ccrs.PlateCarree(),
                        cmap="magma", shading="auto", vmin=5, vmax=60)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.8)
    ax.set_extent(WY_EXTENT, crs=ccrs.PlateCarree())
    add_cities(ax, color="black", edge="white", fontsize=7)
    ax.set_title(f"{name}, valid {str(valid_times[k])[:16]} UTC (+{k+1} h)", fontsize=10)
fig.colorbar(im2, ax=axes, orientation="horizontal", shrink=0.4, pad=0.04, label="Composite reflectivity [dBZ]")
plt.savefig("outputs/wyoming_fc_vs_obs.png", dpi=150); plt.show()""")

md("### Export the Wyoming cut-out")
code(r"""ys, xs = np.where(wy_mask)
sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
ds_out = xr.Dataset(
    {
        **{v: (("lead_time", "y", "x"), goes_fc[:, i, sl[0], sl[1]]) for i, v in enumerate(model_goes.variables)},
        "refc": (("lead_time", "y", "x"), mrms_fc[:, i_refc, sl[0], sl[1]]),
        "refc_obs": (("lead_time", "y", "x"), mrms_obs[:, sl[0], sl[1]]),
    },
    coords=dict(
        lead_time=lead_times[1:], y=model_goes.y[sl[0]], x=model_goes.x[sl[1]],
        lat=(("y", "x"), lat_m[sl]), lon=(("y", "x"), lon_m180[sl]),
        time=np.datetime64(time_ss[0]),
    ),
    attrs=dict(
        title="StormScope 6km_1hr forecast over Wyoming, z500 conditioning from FCN3+InterpModAFNO",
        aiwp_init=str(START_TIME_AIWP), stormscope_init=str(START_TIME_STORMSCOPE), seed=SEED,
        grid="HRRR Lambert conformal, 2x downsampled (6 km)",
    ),
)
ds_out.to_netcdf("outputs/stormscope_aiwp_wyoming.nc")
ds_out""")

md("""## Where to go from here

* **Ensemble**: repeat the batch dimension (`x_goes.repeat(B, ...)`) and/or loop over FCN3 seeds
  with `fcn3.set_rng(seed)` to get a StormScope ensemble driven by an AIWP ensemble; the same
  structure is used by the `foundry_fcn3_stormscope_goes` serving workflow.
* **Swap the AIWP model**: any earth2studio prognostic model that outputs `z500` on a lat/lon grid
  works as the conditioning source. Models without the 73 InterpModAFNO channels can be
  interpolated in time by hand (z500 is smooth at hourly scale).
* **Longer AIWP lead**: increase the gap between `START_TIME_AIWP` and `START_TIME_STORMSCOPE` to
  see how conditioning from a day-old AIWP forecast changes the nowcast.
""")

nb.cells = cells
nbf.write(nb, "aiwp_stormscope_wyoming.ipynb")
print("written", len(cells), "cells")
