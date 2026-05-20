# Van RT-Splatting WebGPU Viewer

A lightweight WebGPU viewer for the RT-Splatting `van` checkpoint, built for iPhone Safari on iOS 26+.

Open the GitHub Pages site in Safari. If the bundled model is not present, tap the folder button and choose `van_mobile.rtsp` from the iPhone Files app.

## Model Asset

The mobile asset generated from the trained checkpoint is:

```text
/home/zhaomy/transparent_gs/RT-Splatting/webgpu-ios-viewer/assets/van_mobile.rtsp
```

Transfer that file to iPhone Files, then open it from the folder button in the viewer.

## Notes

This is a mobile preview renderer. It keeps position, two surfel axes, DC color, opacity, and occupancy from the trained RT-Splatting checkpoint. It does not evaluate RT-Splatting's material/lighting MLP or exact transparency sorting yet.
