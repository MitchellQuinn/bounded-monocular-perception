using System;
using System.Collections.Generic;
using RaccoonBall.SyntheticData.Core;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class StratifiedPlacementCell
    {
        public StratifiedPlacementCell(
            int cellIndex,
            int depthBandIndex,
            int lateralBinIndex,
            int targetSampleCount,
            float depthMinM,
            float depthMaxM,
            float xMinNearM,
            float xMaxNearM,
            float xMinFarM,
            float xMaxFarM)
        {
            CellIndex = cellIndex;
            DepthBandIndex = depthBandIndex;
            LateralBinIndex = lateralBinIndex;
            TargetSampleCount = targetSampleCount;
            DepthMinM = depthMinM;
            DepthMaxM = depthMaxM;
            XMinNearM = xMinNearM;
            XMaxNearM = xMaxNearM;
            XMinFarM = xMinFarM;
            XMaxFarM = xMaxFarM;
        }

        public int CellIndex { get; }
        public int DepthBandIndex { get; }
        public int LateralBinIndex { get; }
        public int TargetSampleCount { get; }

        public float DepthMinM { get; }
        public float DepthMaxM { get; }

        public float XMinNearM { get; }
        public float XMaxNearM { get; }
        public float XMinFarM { get; }
        public float XMaxFarM { get; }
    }

    public sealed class StratifiedPlacementPlan
    {
        public StratifiedPlacementPlan(
            Vector3 planeOriginWorld,
            Vector3 lateralAxisWorld,
            Vector3 depthAxisWorld,
            float movementPlaneY,
            float usableDepthMinM,
            float usableDepthMaxM,
            IReadOnlyList<Vector2> footprintCornersXZ,
            IReadOnlyList<StratifiedPlacementCell> cells,
            int totalSamples)
        {
            PlaneOriginWorld = planeOriginWorld;
            LateralAxisWorld = lateralAxisWorld;
            DepthAxisWorld = depthAxisWorld;
            MovementPlaneY = movementPlaneY;
            UsableDepthMinM = usableDepthMinM;
            UsableDepthMaxM = usableDepthMaxM;
            FootprintCornersXZ = footprintCornersXZ;
            Cells = cells;
            TotalSamples = totalSamples;
        }

        public Vector3 PlaneOriginWorld { get; }
        public Vector3 LateralAxisWorld { get; }
        public Vector3 DepthAxisWorld { get; }
        public float MovementPlaneY { get; }

        public float UsableDepthMinM { get; }
        public float UsableDepthMaxM { get; }
        public IReadOnlyList<Vector2> FootprintCornersXZ { get; }
        public IReadOnlyList<StratifiedPlacementCell> Cells { get; }
        public int TotalSamples { get; }
    }

    public sealed class StratifiedPlacementPlanner
    {
        private const float Epsilon = 0.00001f;

        public StratifiedPlacementPlan BuildPlan(RunConfig config, Camera camera)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (camera == null) throw new ArgumentNullException(nameof(camera));
            if (config.Sweep == null) throw new ArgumentException("RunConfig.Sweep must not be null.");

            SweepSettings sweep = config.Sweep;
            if (sweep.DepthBandCount <= 0) throw new ArgumentException("DepthBandCount must be > 0.");
            if (sweep.LateralBinCount <= 0) throw new ArgumentException("LateralBinCount must be > 0.");

            int totalSamples = ResolveTargetSampleCount(sweep);
            if (totalSamples <= 0) throw new ArgumentException("Total sample count must be > 0.");

            Vector3 planeOriginWorld = new Vector3(camera.transform.position.x, sweep.MovementPlaneY, camera.transform.position.z);
            Vector3 lateralAxisWorld = BuildLateralAxis(camera.transform);
            Vector3 depthAxisWorld = BuildDepthAxis(camera.transform, lateralAxisWorld);

            var footprintCorners = BuildFootprintCorners(camera, sweep.MovementPlaneY, planeOriginWorld, lateralAxisWorld, depthAxisWorld);
            GetDepthRangeFromFootprint(footprintCorners, out float footprintMinDepthM, out float footprintMaxDepthM);

            float usableDepthMinM = footprintMinDepthM;
            float usableDepthMaxM = footprintMaxDepthM;

            if (sweep.UsableDepthMinM > 0f)
            {
                usableDepthMinM = Mathf.Max(usableDepthMinM, sweep.UsableDepthMinM);
            }

            if (sweep.UsableDepthMaxM > 0f)
            {
                usableDepthMaxM = Mathf.Min(usableDepthMaxM, sweep.UsableDepthMaxM);
            }

            if (usableDepthMaxM - usableDepthMinM <= Epsilon)
            {
                throw new InvalidOperationException(
                    $"Usable depth range is invalid. footprint=[{footprintMinDepthM:0.###},{footprintMaxDepthM:0.###}]m, " +
                    $"configured=[{sweep.UsableDepthMinM:0.###},{sweep.UsableDepthMaxM:0.###}]m");
            }

            List<StratifiedPlacementCell> cells = BuildCells(
                footprintCorners,
                usableDepthMinM,
                usableDepthMaxM,
                sweep.DepthBandCount,
                sweep.LateralBinCount,
                totalSamples);

            return new StratifiedPlacementPlan(
                planeOriginWorld,
                lateralAxisWorld,
                depthAxisWorld,
                sweep.MovementPlaneY,
                usableDepthMinM,
                usableDepthMaxM,
                footprintCorners,
                cells,
                totalSamples);
        }

        public Vector3 SamplePositionInCell(StratifiedPlacementPlan plan, StratifiedPlacementCell cell, System.Random rng)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            if (cell == null) throw new ArgumentNullException(nameof(cell));
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            float z = Mathf.Lerp(cell.DepthMinM, cell.DepthMaxM, Next01(rng));
            float zAlpha = cell.DepthMaxM - cell.DepthMinM > Epsilon
                ? Mathf.InverseLerp(cell.DepthMinM, cell.DepthMaxM, z)
                : 0f;

            float xMin = Mathf.Lerp(cell.XMinNearM, cell.XMinFarM, zAlpha);
            float xMax = Mathf.Lerp(cell.XMaxNearM, cell.XMaxFarM, zAlpha);
            if (xMax < xMin)
            {
                (xMin, xMax) = (xMax, xMin);
            }

            float x = Mathf.Lerp(xMin, xMax, Next01(rng));
            Vector3 world = plan.PlaneOriginWorld + (plan.LateralAxisWorld * x) + (plan.DepthAxisWorld * z);
            world.y = plan.MovementPlaneY;
            return world;
        }

        public float ComputeDepthMeters(StratifiedPlacementPlan plan, Vector3 worldPosition)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            Vector3 delta = worldPosition - plan.PlaneOriginWorld;
            return Vector3.Dot(delta, plan.DepthAxisWorld);
        }

        public static int ResolveTargetSampleCount(SweepSettings sweep)
        {
            if (sweep == null) throw new ArgumentNullException(nameof(sweep));

            if (sweep.TotalSamples <= 0)
            {
                throw new ArgumentException("TotalSamples must be > 0.");
            }

            return sweep.TotalSamples;
        }

        private static Vector3 BuildLateralAxis(Transform cameraTransform)
        {
            Vector3 lateral = Vector3.ProjectOnPlane(cameraTransform.right, Vector3.up);
            if (lateral.sqrMagnitude <= Epsilon)
            {
                lateral = Vector3.right;
            }

            lateral.Normalize();
            return lateral;
        }

        private static Vector3 BuildDepthAxis(Transform cameraTransform, Vector3 lateralAxisWorld)
        {
            Vector3 depth = Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up);
            if (depth.sqrMagnitude <= Epsilon)
            {
                depth = Vector3.forward;
            }

            depth.Normalize();

            if (Mathf.Abs(Vector3.Dot(depth, lateralAxisWorld)) > 0.999f)
            {
                depth = Vector3.Cross(Vector3.up, lateralAxisWorld).normalized;
            }

            if (Vector3.Dot(depth, cameraTransform.forward) < 0f)
            {
                depth = -depth;
            }

            return depth;
        }

        private static List<Vector2> BuildFootprintCorners(
            Camera camera,
            float movementPlaneY,
            Vector3 planeOriginWorld,
            Vector3 lateralAxisWorld,
            Vector3 depthAxisWorld)
        {
            var corners = new List<Vector2>(4);
            var movementPlane = new Plane(Vector3.up, new Vector3(0f, movementPlaneY, 0f));

            // Clockwise viewport corners: bottom-left, bottom-right, top-right, top-left.
            ReadCorner(0f, 0f);
            ReadCorner(1f, 0f);
            ReadCorner(1f, 1f);
            ReadCorner(0f, 1f);

            return corners;

            void ReadCorner(float u, float v)
            {
                Ray ray = camera.ViewportPointToRay(new Vector3(u, v, 0f));
                if (!movementPlane.Raycast(ray, out float distance) || distance <= 0f)
                {
                    throw new InvalidOperationException(
                        $"Camera corner ray did not intersect movement plane at uv=({u:0.###},{v:0.###}). " +
                        "Lower the plane Y, increase downward pitch, or clamp usable depth.");
                }

                Vector3 worldPoint = ray.GetPoint(distance);
                Vector3 delta = worldPoint - planeOriginWorld;
                corners.Add(new Vector2(
                    Vector3.Dot(delta, lateralAxisWorld),
                    Vector3.Dot(delta, depthAxisWorld)));
            }
        }

        private static void GetDepthRangeFromFootprint(IReadOnlyList<Vector2> footprintCorners, out float minDepthM, out float maxDepthM)
        {
            minDepthM = float.MaxValue;
            maxDepthM = float.MinValue;

            for (int i = 0; i < footprintCorners.Count; i++)
            {
                float z = footprintCorners[i].y;
                if (z < minDepthM) minDepthM = z;
                if (z > maxDepthM) maxDepthM = z;
            }
        }

        private static List<StratifiedPlacementCell> BuildCells(
            IReadOnlyList<Vector2> footprintCorners,
            float usableDepthMinM,
            float usableDepthMaxM,
            int depthBandCount,
            int lateralBinCount,
            int totalSamples)
        {
            int totalCells = depthBandCount * lateralBinCount;
            int samplesPerCell = totalSamples / totalCells;
            int remainder = totalSamples % totalCells;

            var cells = new List<StratifiedPlacementCell>(totalCells);
            int cellIndex = 0;

            for (int depthBandIndex = 0; depthBandIndex < depthBandCount; depthBandIndex++)
            {
                float zMin = Mathf.Lerp(usableDepthMinM, usableDepthMaxM, depthBandIndex / (float)depthBandCount);
                float zMax = Mathf.Lerp(usableDepthMinM, usableDepthMaxM, (depthBandIndex + 1) / (float)depthBandCount);

                float nearProbeZ = BuildBandProbeDepth(zMin, zMax, true);
                float farProbeZ = BuildBandProbeDepth(zMin, zMax, false);

                if (!TryGetLateralRangeAtDepth(footprintCorners, nearProbeZ, out float xNearMin, out float xNearMax))
                {
                    throw new InvalidOperationException($"Failed to compute lateral range at near depth z={nearProbeZ:0.###}.");
                }

                if (!TryGetLateralRangeAtDepth(footprintCorners, farProbeZ, out float xFarMin, out float xFarMax))
                {
                    throw new InvalidOperationException($"Failed to compute lateral range at far depth z={farProbeZ:0.###}.");
                }

                for (int lateralBinIndex = 0; lateralBinIndex < lateralBinCount; lateralBinIndex++)
                {
                    float tMin = lateralBinIndex / (float)lateralBinCount;
                    float tMax = (lateralBinIndex + 1) / (float)lateralBinCount;

                    float xMinNear = Mathf.Lerp(xNearMin, xNearMax, tMin);
                    float xMaxNear = Mathf.Lerp(xNearMin, xNearMax, tMax);
                    float xMinFar = Mathf.Lerp(xFarMin, xFarMax, tMin);
                    float xMaxFar = Mathf.Lerp(xFarMin, xFarMax, tMax);

                    int targetSampleCount = samplesPerCell + (cellIndex < remainder ? 1 : 0);
                    cells.Add(new StratifiedPlacementCell(
                        cellIndex,
                        depthBandIndex,
                        lateralBinIndex,
                        targetSampleCount,
                        zMin,
                        zMax,
                        xMinNear,
                        xMaxNear,
                        xMinFar,
                        xMaxFar));

                    cellIndex++;
                }
            }

            return cells;
        }

        private static float BuildBandProbeDepth(float zMin, float zMax, bool near)
        {
            float width = Mathf.Abs(zMax - zMin);
            if (width <= Epsilon)
            {
                return 0.5f * (zMin + zMax);
            }

            float offset = Mathf.Min(0.0001f, width * 0.25f);
            return near ? zMin + offset : zMax - offset;
        }

        private static bool TryGetLateralRangeAtDepth(IReadOnlyList<Vector2> polygon, float depthZ, out float minX, out float maxX)
        {
            var intersections = new List<float>(8);

            for (int i = 0; i < polygon.Count; i++)
            {
                Vector2 a = polygon[i];
                Vector2 b = polygon[(i + 1) % polygon.Count];

                if (Mathf.Abs(b.y - a.y) <= Epsilon)
                {
                    if (Mathf.Abs(depthZ - a.y) <= 0.0002f)
                    {
                        intersections.Add(a.x);
                        intersections.Add(b.x);
                    }

                    continue;
                }

                float t = (depthZ - a.y) / (b.y - a.y);
                if (t < -0.0001f || t > 1.0001f)
                {
                    continue;
                }

                intersections.Add(Mathf.Lerp(a.x, b.x, Mathf.Clamp01(t)));
            }

            if (intersections.Count < 2)
            {
                minX = 0f;
                maxX = 0f;
                return false;
            }

            intersections.Sort();

            float dedupeEpsilon = 0.0002f;
            var unique = new List<float>(intersections.Count);
            for (int i = 0; i < intersections.Count; i++)
            {
                if (unique.Count == 0 || Mathf.Abs(unique[unique.Count - 1] - intersections[i]) > dedupeEpsilon)
                {
                    unique.Add(intersections[i]);
                }
            }

            if (unique.Count < 2)
            {
                minX = 0f;
                maxX = 0f;
                return false;
            }

            minX = unique[0];
            maxX = unique[unique.Count - 1];
            return maxX - minX > Epsilon;
        }

        private static float Next01(System.Random rng)
        {
            return (float)rng.NextDouble();
        }
    }
}
