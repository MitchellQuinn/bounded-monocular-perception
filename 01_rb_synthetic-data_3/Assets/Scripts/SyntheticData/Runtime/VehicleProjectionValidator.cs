using System;
using RaccoonBall.SyntheticData.Core;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class VehicleProjectionValidator
    {
        private readonly Camera _camera;
        private readonly Renderer[] _renderers;
        private readonly CaptureSettings _captureSettings;
        private readonly SweepSettings _sweepSettings;
        private readonly Vector3[] _boundsCorners = new Vector3[8];

        public VehicleProjectionValidator(
            Camera camera,
            Transform vehicleRoot,
            CaptureSettings captureSettings,
            SweepSettings sweepSettings)
        {
            _camera = camera ? camera : throw new ArgumentNullException(nameof(camera));
            if (vehicleRoot == null) throw new ArgumentNullException(nameof(vehicleRoot));
            _captureSettings = captureSettings ?? throw new ArgumentNullException(nameof(captureSettings));
            _sweepSettings = sweepSettings ?? throw new ArgumentNullException(nameof(sweepSettings));

            _renderers = vehicleRoot.GetComponentsInChildren<Renderer>(true);
            if (_renderers.Length == 0)
            {
                throw new InvalidOperationException("Vehicle root has no Renderer components. Cannot validate projected footprint.");
            }
        }

        public bool IsPlacementValid(out string rejectionReason)
        {
            bool hasProjectedPoint = false;
            float minViewportX = float.MaxValue;
            float maxViewportX = float.MinValue;
            float minViewportY = float.MaxValue;
            float maxViewportY = float.MinValue;

            for (int rendererIndex = 0; rendererIndex < _renderers.Length; rendererIndex++)
            {
                Renderer renderer = _renderers[rendererIndex];
                if (renderer == null || !renderer.enabled || !renderer.gameObject.activeInHierarchy)
                {
                    continue;
                }

                GetBoundsCorners(renderer.bounds, _boundsCorners);
                for (int i = 0; i < _boundsCorners.Length; i++)
                {
                    Vector3 viewport = _camera.WorldToViewportPoint(_boundsCorners[i]);
                    if (viewport.z <= 0f)
                    {
                        rejectionReason = "Projected bounds fall behind the camera.";
                        return false;
                    }

                    if (viewport.x < minViewportX) minViewportX = viewport.x;
                    if (viewport.x > maxViewportX) maxViewportX = viewport.x;
                    if (viewport.y < minViewportY) minViewportY = viewport.y;
                    if (viewport.y > maxViewportY) maxViewportY = viewport.y;
                    hasProjectedPoint = true;
                }
            }

            if (!hasProjectedPoint)
            {
                rejectionReason = "No enabled renderer bounds were available for projection.";
                return false;
            }

            if (minViewportX < 0f || maxViewportX > 1f || minViewportY < 0f || maxViewportY > 1f)
            {
                rejectionReason = "Projected defender clips the camera frame.";
                return false;
            }

            float marginX = _sweepSettings.EdgeMarginPx / _captureSettings.ImageWidthPx;
            float marginY = _sweepSettings.EdgeMarginPx / _captureSettings.ImageHeightPx;
            if (minViewportX < marginX || maxViewportX > 1f - marginX || minViewportY < marginY || maxViewportY > 1f - marginY)
            {
                rejectionReason = "Projected defender violates configured edge margin.";
                return false;
            }

            float projectedWidthPx = (maxViewportX - minViewportX) * _captureSettings.ImageWidthPx;
            float projectedHeightPx = (maxViewportY - minViewportY) * _captureSettings.ImageHeightPx;
            float projectedAreaPx = projectedWidthPx * projectedHeightPx;

            if (projectedWidthPx < _sweepSettings.MinProjectedWidthPx)
            {
                rejectionReason = $"Projected width too small ({projectedWidthPx:0.##} px).";
                return false;
            }

            if (_sweepSettings.MaxProjectedWidthPx > 0f && projectedWidthPx > _sweepSettings.MaxProjectedWidthPx)
            {
                rejectionReason = $"Projected width too large ({projectedWidthPx:0.##} px).";
                return false;
            }

            if (projectedHeightPx < _sweepSettings.MinProjectedHeightPx)
            {
                rejectionReason = $"Projected height too small ({projectedHeightPx:0.##} px).";
                return false;
            }

            if (_sweepSettings.MaxProjectedHeightPx > 0f && projectedHeightPx > _sweepSettings.MaxProjectedHeightPx)
            {
                rejectionReason = $"Projected height too large ({projectedHeightPx:0.##} px).";
                return false;
            }

            if (projectedAreaPx < _sweepSettings.MinProjectedAreaPx)
            {
                rejectionReason = $"Projected area too small ({projectedAreaPx:0.##} px^2).";
                return false;
            }

            if (_sweepSettings.MaxProjectedAreaPx > 0f && projectedAreaPx > _sweepSettings.MaxProjectedAreaPx)
            {
                rejectionReason = $"Projected area too large ({projectedAreaPx:0.##} px^2).";
                return false;
            }

            rejectionReason = string.Empty;
            return true;
        }

        private static void GetBoundsCorners(Bounds bounds, Vector3[] outputCorners)
        {
            Vector3 min = bounds.min;
            Vector3 max = bounds.max;

            outputCorners[0] = new Vector3(min.x, min.y, min.z);
            outputCorners[1] = new Vector3(max.x, min.y, min.z);
            outputCorners[2] = new Vector3(min.x, max.y, min.z);
            outputCorners[3] = new Vector3(max.x, max.y, min.z);
            outputCorners[4] = new Vector3(min.x, min.y, max.z);
            outputCorners[5] = new Vector3(max.x, min.y, max.z);
            outputCorners[6] = new Vector3(min.x, max.y, max.z);
            outputCorners[7] = new Vector3(max.x, max.y, max.z);
        }
    }
}
