using System;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class CoordinateConvention
    {
        public Vector3 CameraPosition = new Vector3(0f, 1.5f, 0f);
        public Vector3 CameraRotationEulerDeg = new Vector3(27.5f, 0f, 0f);

        public Vector3 VehicleBasePositionAtSweepStart = new Vector3(0f, 0.5f, 2f);
        public Vector3 VehicleBaseRotationEulerDeg = new Vector3(0f, 180f, 0f);

        public string SweepAxisName = "Z";
        public string WorldUpAxisName = "Y";
        public string MovingObjectName = "Vehicle";
        public string FixedObjectName = "Camera";
        public string DistanceDefinition = "Euclidean distance from camera position to vehicle root position";
    }
}
