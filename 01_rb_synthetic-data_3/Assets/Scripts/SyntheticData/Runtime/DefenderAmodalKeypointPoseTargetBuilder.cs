using System;
using RaccoonBall.SyntheticData.Core;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class DefenderAmodalKeypointPoseTargetBuilder
    {
        private const int RequiredKeypointCount = 10;
        private const int RequiredCoordinateWidth = 3;
        private const string ExpectedSchemaVersion = "0.1.0";
        private const string ExpectedSchemaHash = "157d0224ec463338b79855fb26c58e1b01242c7d83c016c122ba2a3135e4eb93";
        private const string ExpectedSchemaPath = "03_rb-training-v2.0/schemas/defender_keypoint_schema.json";
        private const string ExpectedCoordinateSpace = "camera_space_3d";
        private const string ExpectedFlatteningOrder = "keypoint_major_xyz";

        private readonly Camera _camera;
        private readonly Transform _vehicleRoot;
        private readonly DefenderAmodalKeypointPoseTargetSettings _settings;
        private readonly AxisMapping _axes;
        private readonly LocalHull _hull;
        private readonly Vector3[] _keypointsLocal;
        private readonly Vector3 _centerLocal;

        private readonly struct AxisMapping
        {
            public readonly int Right;
            public readonly int Up;
            public readonly int Forward;
            public readonly int ForwardSign;

            public AxisMapping(int right, int up, int forward, int forwardSign)
            {
                Right = right;
                Up = up;
                Forward = forward;
                ForwardSign = forwardSign >= 0 ? 1 : -1;
            }
        }

        private readonly struct LocalHull
        {
            public readonly float Left;
            public readonly float Right;
            public readonly float Bottom;
            public readonly float Body;
            public readonly float Roof;
            public readonly float Rear;
            public readonly float Front;
            public readonly float ScreenTop;

            public LocalHull(
                float left,
                float right,
                float bottom,
                float body,
                float roof,
                float rear,
                float front,
                float screenTop)
            {
                Left = left;
                Right = right;
                Bottom = bottom;
                Body = body;
                Roof = roof;
                Rear = rear;
                Front = front;
                ScreenTop = screenTop;
            }
        }

        public DefenderAmodalKeypointPoseTargetBuilder(
            Camera camera,
            Transform vehicleRoot,
            DefenderAmodalKeypointPoseTargetSettings settings)
        {
            _camera = camera ? camera : throw new ArgumentNullException(nameof(camera));
            _vehicleRoot = vehicleRoot ? vehicleRoot : throw new ArgumentNullException(nameof(vehicleRoot));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));

            ValidateSettingsForRun(_settings, _vehicleRoot);
            _axes = BuildAxisMapping(_settings);
            Bounds localBounds = ComputeVehicleLocalMeshBounds(_vehicleRoot);
            _hull = BuildLocalHull(localBounds, _axes, _settings);
            _keypointsLocal = BuildKeypointsLocal(_hull, _axes);
            _centerLocal = localBounds.center;
        }

        public DefenderAmodalKeypointPoseTargets Build()
        {
            Vector3 centerWorld = _vehicleRoot.TransformPoint(_centerLocal);
            Vector3 centerCameraSpace = _camera.transform.InverseTransformPoint(centerWorld);

            var keypoints = new Vector3[RequiredKeypointCount];
            var visibility = new bool[RequiredKeypointCount];

            for (int i = 0; i < RequiredKeypointCount; i++)
            {
                Vector3 keypointWorld = _vehicleRoot.TransformPoint(_keypointsLocal[i]);
                keypoints[i] = _camera.transform.InverseTransformPoint(keypointWorld);
                visibility[i] = IsKeypointVisible(i, keypointWorld);
            }

            return new DefenderAmodalKeypointPoseTargets(centerCameraSpace, keypoints, visibility);
        }

        public static void ValidateSettingsForRun(
            DefenderAmodalKeypointPoseTargetSettings settings,
            Transform vehicleRoot)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (vehicleRoot == null) throw new ArgumentNullException(nameof(vehicleRoot));

            settings.EnsureDefaults();
            ValidateSchema(settings.Schema);
            ValidateKeypoints(settings);
            _ = BuildAxisMapping(settings);

            if (!IsFinite(settings.BodyHeightFractionOfBounds) ||
                settings.BodyHeightFractionOfBounds <= 0f ||
                settings.BodyHeightFractionOfBounds >= 1f)
            {
                throw new ArgumentException("Defender BodyHeightFractionOfBounds must be finite and between 0 and 1.");
            }

            if (!IsFinite(settings.ScreenTopForwardFractionFromRear) ||
                settings.ScreenTopForwardFractionFromRear <= 0f ||
                settings.ScreenTopForwardFractionFromRear >= 1f)
            {
                throw new ArgumentException("Defender ScreenTopForwardFractionFromRear must be finite and between 0 and 1.");
            }

            if (settings.VisibilitySurfaceToleranceM < 0f || !IsFinite(settings.VisibilitySurfaceToleranceM))
            {
                throw new ArgumentException("Defender visibility surface tolerance must be a finite value >= 0.");
            }

            if (settings.RequireVisibilityColliders)
            {
                Collider[] colliders = vehicleRoot.GetComponentsInChildren<Collider>(true);
                if (colliders == null || colliders.Length == 0)
                {
                    throw new InvalidOperationException(
                        "Defender keypoint occlusion checks require at least one Collider under the vehicle root. " +
                        "Disable RequireVisibilityColliders to use hull self-visibility only.");
                }
            }
        }

        private bool IsKeypointVisible(int keypointIndex, Vector3 keypointWorld)
        {
            Vector3 viewport = _camera.WorldToViewportPoint(keypointWorld);
            if (viewport.z <= 0f || viewport.x < 0f || viewport.x > 1f || viewport.y < 0f || viewport.y > 1f)
            {
                return false;
            }

            if (!IsHullDatumVisibleFromCamera(keypointIndex))
            {
                return false;
            }

            if (!_settings.RequireVisibilityColliders)
            {
                return true;
            }

            Vector3 cameraPosition = _camera.transform.position;
            Vector3 rayVector = keypointWorld - cameraPosition;
            float targetDistance = rayVector.magnitude;
            if (targetDistance <= 0f)
            {
                return true;
            }

            Vector3 direction = rayVector / targetDistance;
            int layerMask = _settings.VisibilityLayerMask.value;
            bool hitSomething = Physics.Raycast(
                cameraPosition,
                direction,
                out RaycastHit hit,
                targetDistance + Mathf.Max(0f, _settings.VisibilitySurfaceToleranceM),
                layerMask,
                QueryTriggerInteraction.Ignore);

            if (!hitSomething)
            {
                return true;
            }

            return IsVehicleTransform(hit.transform);
        }

        private bool IsHullDatumVisibleFromCamera(int keypointIndex)
        {
            Vector3 cameraLocal = _vehicleRoot.InverseTransformPoint(_camera.transform.position);
            float right = Component(cameraLocal, _axes.Right);
            float up = Component(cameraLocal, _axes.Up);
            float forward = Component(cameraLocal, _axes.Forward);

            bool seesLeft = right <= _hull.Left;
            bool seesRight = right >= _hull.Right;
            bool seesBottom = up <= _hull.Bottom;
            bool seesRoof = up >= _hull.Roof;
            bool seesBodyTop = up >= _hull.Body;
            bool seesRear = _axes.ForwardSign > 0 ? forward <= _hull.Rear : forward >= _hull.Rear;
            bool seesFront = _axes.ForwardSign > 0 ? forward >= _hull.Front : forward <= _hull.Front;
            bool seesScreenFront = _axes.ForwardSign > 0 ? forward >= _hull.ScreenTop : forward <= _hull.ScreenTop;

            switch (keypointIndex)
            {
                case 0: return seesRear || seesLeft || seesBottom;
                case 1: return seesRear || seesRight || seesBottom;
                case 2: return seesRear || seesLeft || seesRoof;
                case 3: return seesRear || seesRight || seesRoof;
                case 4: return seesFront || seesLeft || seesBottom;
                case 5: return seesFront || seesRight || seesBottom;
                case 6: return seesFront || seesLeft || seesBodyTop;
                case 7: return seesFront || seesRight || seesBodyTop;
                case 8: return seesScreenFront || seesLeft || seesRoof;
                case 9: return seesScreenFront || seesRight || seesRoof;
                default: return false;
            }
        }

        private bool IsVehicleTransform(Transform candidate)
        {
            return candidate == _vehicleRoot || (candidate != null && candidate.IsChildOf(_vehicleRoot));
        }

        private static Bounds ComputeVehicleLocalMeshBounds(Transform vehicleRoot)
        {
            MeshFilter[] meshFilters = vehicleRoot.GetComponentsInChildren<MeshFilter>(true);
            bool hasPoint = false;
            Bounds bounds = default;

            for (int i = 0; i < meshFilters.Length; i++)
            {
                MeshFilter filter = meshFilters[i];
                if (filter == null || filter.sharedMesh == null)
                {
                    continue;
                }

                Bounds meshBounds = filter.sharedMesh.bounds;
                var corners = new[]
                {
                    new Vector3(meshBounds.min.x, meshBounds.min.y, meshBounds.min.z),
                    new Vector3(meshBounds.max.x, meshBounds.min.y, meshBounds.min.z),
                    new Vector3(meshBounds.min.x, meshBounds.max.y, meshBounds.min.z),
                    new Vector3(meshBounds.max.x, meshBounds.max.y, meshBounds.min.z),
                    new Vector3(meshBounds.min.x, meshBounds.min.y, meshBounds.max.z),
                    new Vector3(meshBounds.max.x, meshBounds.min.y, meshBounds.max.z),
                    new Vector3(meshBounds.min.x, meshBounds.max.y, meshBounds.max.z),
                    new Vector3(meshBounds.max.x, meshBounds.max.y, meshBounds.max.z),
                };

                for (int c = 0; c < corners.Length; c++)
                {
                    Vector3 local = vehicleRoot.InverseTransformPoint(filter.transform.TransformPoint(corners[c]));
                    if (!hasPoint)
                    {
                        bounds = new Bounds(local, Vector3.zero);
                        hasPoint = true;
                    }
                    else
                    {
                        bounds.Encapsulate(local);
                    }
                }
            }

            if (!hasPoint)
            {
                throw new InvalidOperationException(
                    "Cannot derive Defender keypoint hull: no MeshFilter/sharedMesh geometry found under the vehicle root.");
            }

            if (bounds.size.sqrMagnitude <= 1e-10f)
            {
                throw new InvalidOperationException("Cannot derive Defender keypoint hull from degenerate vehicle mesh bounds.");
            }

            return bounds;
        }

        private static LocalHull BuildLocalHull(
            Bounds bounds,
            AxisMapping axes,
            DefenderAmodalKeypointPoseTargetSettings settings)
        {
            float rightMin = Component(bounds.min, axes.Right);
            float rightMax = Component(bounds.max, axes.Right);
            float upMin = Component(bounds.min, axes.Up);
            float upMax = Component(bounds.max, axes.Up);
            float forwardMin = Component(bounds.min, axes.Forward);
            float forwardMax = Component(bounds.max, axes.Forward);
            float rear = axes.ForwardSign > 0 ? forwardMin : forwardMax;
            float front = axes.ForwardSign > 0 ? forwardMax : forwardMin;
            float body = Mathf.Lerp(upMin, upMax, settings.BodyHeightFractionOfBounds);
            float screenTop = Mathf.Lerp(rear, front, settings.ScreenTopForwardFractionFromRear);

            return new LocalHull(
                rightMin,
                rightMax,
                upMin,
                body,
                upMax,
                rear,
                front,
                screenTop);
        }

        private static Vector3[] BuildKeypointsLocal(LocalHull hull, AxisMapping axes)
        {
            return new[]
            {
                Point(axes, hull.Left, hull.Bottom, hull.Rear),
                Point(axes, hull.Right, hull.Bottom, hull.Rear),
                Point(axes, hull.Left, hull.Roof, hull.Rear),
                Point(axes, hull.Right, hull.Roof, hull.Rear),
                Point(axes, hull.Left, hull.Bottom, hull.Front),
                Point(axes, hull.Right, hull.Bottom, hull.Front),
                Point(axes, hull.Left, hull.Body, hull.Front),
                Point(axes, hull.Right, hull.Body, hull.Front),
                Point(axes, hull.Left, hull.Roof, hull.ScreenTop),
                Point(axes, hull.Right, hull.Roof, hull.ScreenTop),
            };
        }

        private static Vector3 Point(AxisMapping axes, float right, float up, float forward)
        {
            Vector3 result = Vector3.zero;
            result = WithComponent(result, axes.Right, right);
            result = WithComponent(result, axes.Up, up);
            result = WithComponent(result, axes.Forward, forward);
            return result;
        }

        private static void ValidateSchema(DefenderKeypointSchemaMetadata schema)
        {
            if (schema == null) throw new ArgumentException("Defender keypoint schema metadata must not be null.");

            RequireEqual(schema.SchemaVersion, ExpectedSchemaVersion, "schema version");
            RequireEqual(schema.SchemaHash, ExpectedSchemaHash, "schema hash");
            RequireEqual(schema.SchemaPath, ExpectedSchemaPath, "schema path");
            RequireEqual(schema.CoordinateSpace, ExpectedCoordinateSpace, "coordinate space");
            RequireEqual(schema.FlatteningOrder, ExpectedFlatteningOrder, "flattening order");

            if (schema.NumKeypoints != RequiredKeypointCount)
            {
                throw new ArgumentException($"Defender keypoint schema num_keypoints must be {RequiredKeypointCount}.");
            }

            if (schema.CoordinateWidth != RequiredCoordinateWidth)
            {
                throw new ArgumentException($"Defender keypoint schema coordinate_width must be {RequiredCoordinateWidth}.");
            }

            if (!schema.TrainingAllowed || !string.Equals(schema.SchemaStatus, "active", StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Defender amodal keypoint pose schema must be active before generation.");
            }
        }

        private static void ValidateKeypoints(DefenderAmodalKeypointPoseTargetSettings settings)
        {
            string[] expectedNames =
            {
                "rear_bottom_left",
                "rear_bottom_right",
                "rear_roof_left",
                "rear_roof_right",
                "front_bottom_left",
                "front_bottom_right",
                "front_body_left",
                "front_body_right",
                "screen_top_left",
                "screen_top_right",
            };

            if (settings.Keypoints == null || settings.Keypoints.Length != RequiredKeypointCount)
            {
                throw new ArgumentException($"Defender keypoint definitions must contain exactly {RequiredKeypointCount} entries.");
            }

            for (int i = 0; i < RequiredKeypointCount; i++)
            {
                DefenderAmodalKeypointDefinition keypoint = settings.Keypoints[i];
                if (keypoint == null)
                {
                    throw new ArgumentException($"Defender keypoint definition {i} must not be null.");
                }

                if (keypoint.Index != i)
                {
                    throw new ArgumentException($"Defender keypoint definition at slot {i} must have Index={i}.");
                }

                if (!string.Equals(keypoint.Name, expectedNames[i], StringComparison.Ordinal))
                {
                    throw new ArgumentException(
                        $"Defender keypoint {i} name mismatch; expected '{expectedNames[i]}', got '{keypoint.Name ?? string.Empty}'.");
                }

                if (!string.Equals(keypoint.Status, "defined", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException($"Defender keypoint {i} ('{keypoint.Name}') must be marked defined.");
                }
            }
        }

        private static AxisMapping BuildAxisMapping(DefenderAmodalKeypointPoseTargetSettings settings)
        {
            int right = ParseAxis(settings.RightAxisName, nameof(settings.RightAxisName));
            int up = ParseAxis(settings.UpAxisName, nameof(settings.UpAxisName));
            int forward = ParseAxis(settings.ForwardAxisName, nameof(settings.ForwardAxisName));
            if (right == up || right == forward || up == forward)
            {
                throw new ArgumentException("Defender keypoint axis mapping must use three distinct axes.");
            }

            if (settings.ForwardAxisSign != 1 && settings.ForwardAxisSign != -1)
            {
                throw new ArgumentException("Defender ForwardAxisSign must be 1 or -1.");
            }

            return new AxisMapping(right, up, forward, settings.ForwardAxisSign);
        }

        private static int ParseAxis(string axisName, string label)
        {
            string value = string.IsNullOrWhiteSpace(axisName) ? string.Empty : axisName.Trim().ToUpperInvariant();
            if (value == "X") return 0;
            if (value == "Y") return 1;
            if (value == "Z") return 2;
            throw new ArgumentException($"{label} must be one of X, Y, or Z.");
        }

        private static float Component(Vector3 value, int axis)
        {
            switch (axis)
            {
                case 0: return value.x;
                case 1: return value.y;
                case 2: return value.z;
                default: throw new ArgumentOutOfRangeException(nameof(axis));
            }
        }

        private static Vector3 WithComponent(Vector3 value, int axis, float component)
        {
            switch (axis)
            {
                case 0:
                    value.x = component;
                    return value;
                case 1:
                    value.y = component;
                    return value;
                case 2:
                    value.z = component;
                    return value;
                default:
                    throw new ArgumentOutOfRangeException(nameof(axis));
            }
        }

        private static void RequireEqual(string actual, string expected, string label)
        {
            if (!string.Equals(actual, expected, StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Defender keypoint {label} mismatch; expected '{expected}', got '{actual ?? string.Empty}'.");
            }
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
