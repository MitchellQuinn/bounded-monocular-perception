using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class RunConfig
    {
        public string RunId = "run_2026-03-26_def90_v001";
        public string DatasetName = "def90_synth_v001";
        public OutputSettings Output = new OutputSettings();
        public CaptureSettings Capture = new CaptureSettings();
        public SweepSettings Sweep = new SweepSettings();
        public CoordinateConvention CoordinateConvention = new CoordinateConvention();
        public JitterPolicy JitterPolicy = new JitterPolicy();
        public int RandomSeed = 123456789;
        public string Notes = string.Empty;
    }
}
