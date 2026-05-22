# Unity AR0234 Analytic Pinhole Calibration

This folder mirrors the ChArUco calibration artifact shape used by the AR0234 run at `charuco-calibration/calibration_runs/260519-1501_calibio_charuco_30mm_a4`.

The Unity camera is derived analytically from `01_rb_synthetic-data_3/Assets/Scenes/rb_synthetic-data_1.unity`: focal length 3.56 mm, sensor size 5.76 x 3.60 mm, capture size 1920 x 1200, and zero lens shift. That gives `fx = fy = 1186.6666666667 px`, `cx = 960 px`, `cy = 600 px`, and zero OpenCV distortion coefficients.

Use this as the target camera model when building the AR0234-to-Unity image remap. This is not a ChArUco solve; `accepted_frame_count`, `used_frame_count`, and reprojection errors are zero because Unity is represented as an ideal physical-camera pinhole model.
