namespace RaccoonBall.SyntheticData.Core
{
    public sealed class ManifestRow
    {
        public const string PlacementBinIdColumnName = "placement_bin_id";

        public string RunId;
        public string SampleId;
        public int FrameIndex;
        public string ImageFilename;

        public int PlacementBinId;
        public int PositionStepIndex;
        public int SampleAtPositionIndex;

        public float BasePosXM;
        public float BasePosYM;
        public float BasePosZM;

        public float BaseRotXDeg;
        public float BaseRotYDeg;
        public float BaseRotZDeg;

        public float JitterPosXM;
        public float JitterPosYM;
        public float JitterPosZM;

        public float JitterRotXDeg;
        public float JitterRotYDeg;
        public float JitterRotZDeg;

        public float FinalPosXM;
        public float FinalPosYM;
        public float FinalPosZM;

        public float FinalRotXDeg;
        public float FinalRotYDeg;
        public float FinalRotZDeg;

        public float DistanceM;

        public int ImageWidthPx;
        public int ImageHeightPx;

        public bool CaptureSuccess;
        public string ErrorMessage;
    }
}
