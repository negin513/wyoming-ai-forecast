# AI Weather Forecasting over Wyoming

Hands-on notebooks for the University of Wyoming training through Earth2Studio. 

| Notebook | What it does |
|---|---|
| [`01_wyoming_ai_forecast.ipynb`](01_wyoming_ai_forecast.ipynb) | 5-day deterministic **FourCastNet (FCN)** forecast from GFS analysis, cropped to Wyoming (`t2m`, `u10m`, `v10m`, `msl`). |
| [`02_stormscope_wyoming_demo.ipynb`](02_stormscope_wyoming_demo.ipynb) | Nowcasting with **StormScope** over Wyoming. |


## References

- [Documentation](https://nvidia.github.io/earth2studio/) · [Install guide](https://nvidia.github.io/earth2studio/main/userguide/about/install/)
- [Prognostic model catalog](https://nvidia.github.io/earth2studio/main/modules/models_px/)
- [Examples gallery](https://nvidia.github.io/earth2studio/main/examples/) · [StormScope GOES/MRMS example](https://nvidia.github.io/earth2studio/main/examples/04_nowcasting/03_stormscope_goes_example/)
- [Source code](https://github.com/NVIDIA/earth2studio) · [Issues](https://github.com/NVIDIA/earth2studio/issues)

- **StormScope**: Pathak et al., *Learning Accurate Storm-Scale Evolution from Observations*, [arXiv:2601.17268](https://arxiv.org/abs/2601.17268)