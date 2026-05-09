using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class CameraJitterPolicy
    {
        public float PosYMinM = -0.01f;
        public float PosYMaxM = 0.01f;
        public float RotXMinDeg = -2f;
        public float RotXMaxDeg = 2f;
    }

    [Serializable]
    public sealed class VehicleJitterPolicy
    {
        public float RotYMinDeg = 0f;
        public float RotYMaxDeg = 0f;
    }
}
