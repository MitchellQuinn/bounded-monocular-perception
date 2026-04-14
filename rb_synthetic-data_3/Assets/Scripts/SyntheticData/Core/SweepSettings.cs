using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class SweepSettings
    {
        public int TotalSamples = 48096;

        public float MovementPlaneY = 0.5f;

        public int DepthBandCount = 12;
        public int LateralBinCount = 10;

        // When > 0, these clamp the camera-visible footprint depth range.
        public float UsableDepthMinM = 0f;
        public float UsableDepthMaxM = 0f;

        public float EdgeMarginPx = 12f;
        public float MinProjectedWidthPx = 64f;
        public float MaxProjectedWidthPx = 0f;
        public float MinProjectedHeightPx = 48f;
        public float MaxProjectedHeightPx = 0f;
        public float MinProjectedAreaPx = 4096f;
        public float MaxProjectedAreaPx = 0f;

        public int MaxAttemptsPerSample = 200;
        public int MaxAttemptsPerCell = 200000;

        public int FeasibilityProbeAttemptsPerCell = 2000;

        public int AcceptanceProbeAttemptsPerCell = 2000;
        public int MinSamplesPerFeasibleCell = 4;
        public float MinAcceptanceRateForWeight = 0.01f;
        public float AcceptanceWeightExponent = 1f;
        public bool LogCellAcceptanceStats = false;
    }
}
