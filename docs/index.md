# Wyoming AI weather forecast

Notebooks that run NVIDIA Earth2Studio AI weather models and look at the
result, for University of Wyoming training. (DLI additions)

| Notebook | What it does |
|---|---|
| [Wyoming AI forecast](01_wyoming_ai_forecast.ipynb) | 5-day deterministic FourCastNet (FCN) forecast from GFS analysis, cropped to Wyoming. |
| [StormScope over Wyoming](02_stormscope_wyoming_demo.ipynb) | FourCastNet 3 (hourly, via InterpModAFNO) providing `z500` conditioning to **StormScope** GOES + MRMS nowcasting on the CONUS grid, analysed and verified over Wyoming. Executed, with outputs embedded. |
