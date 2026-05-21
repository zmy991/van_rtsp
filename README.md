# Van RT-Splatting WebGPU Viewer

A lightweight WebGPU viewer for the RT-Splatting `van` checkpoint, built for iPhone Safari on iOS 26+.

This version supports the richer `RTSPGS2` mobile asset format:

- degree-3 spherical harmonics for view-dependent color
- occupancy, opacity, transmissivity, roughness, reflectance, and inside-mask fields
- 4D learned feature vector used as a lightweight specular tint
- optional CPU depth sorting with an indexed WebGPU draw order buffer

It still does not exactly reproduce RT-Splatting's CUDA rasterizer, SphMip texture lookup, or PyTorch `light_mlp`; the browser shader uses a mobile-friendly approximation for reflection/specular.

## Build the full mobile asset

```bash
/home/zhaomy/miniconda3/envs/rtsplat/bin/python \
  webgpu-ios-viewer/tools/convert_rt_splat_ply.py \
  --input output/van/point_cloud/iteration_61000/point_cloud.ply \
  --output webgpu-ios-viewer/assets/van_mobile_full.rtsp \
  --asset-format full \
  --max-splats 180000 \
  --crop-center -0.5 -1.0 0.5 \
  --crop-radius 5.0 \
  --env-center -0.5 -1.0 0.5 \
  --env-radius 2.5
```

Open the deployed site in Safari. If the bundled model is not present, tap the folder button and choose `van_mobile_full.rtsp` from the iPhone Files app.
