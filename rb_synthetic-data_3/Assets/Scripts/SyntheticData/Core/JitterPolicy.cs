using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class JitterPolicy
    {
        public float PosXMinM = 0f;
        public float PosXMaxM = 0f;
        public float PosZMinM = 0f;
        public float PosZMaxM = 0f;

        public float RotYMinDeg = 0f;
        public float RotYMaxDeg = 0f;
    }
}
