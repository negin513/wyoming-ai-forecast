# AI Weather Forecasting over Wyoming

Hands-on notebooks for the University of Wyoming training through Earth2Studio. 

| Notebook | What it does |
|---|---|
| [`01_wyoming_ai_forecast.ipynb`](01_wyoming_ai_forecast.ipynb) | 5-day deterministic **FourCastNet (FCN)** forecast from GFS analysis, cropped to Wyoming (`t2m`, `u10m`, `v10m`, `msl`). |
| [`02_stormscope_wyoming_demo.ipynb`](02_stormscope_wyoming_demo.ipynb) | Nowcasting with **StormScope** over Wyoming. |


## 📚 References

* 📄 **Reference**
  * [StormScope (arXiv:2601.17268)](https://arxiv.org/html/2601.17268v1) — the paper behind the observation-to-observation nowcasting model used in this notebook
  * [Earth2Studio docs](https://nvidia.github.io/earth2studio/) — the inference framework providing `StormScopeGOES`/`StormScopeMRMS`, and the GOES/MRMS/GLM data sources used throughout
  * [Earth2Studio (code)](https://github.com/NVIDIA/earth2studio) — source repository
* 🙋 **Ask for help**
  * [Earth2Studio GitHub issues](https://github.com/NVIDIA/earth2studio/issues) for bug reports and feature requests
