using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class RunMetadata
    {
        public string RunId;
        public string DatasetName;

        public OutputSettings Output;
        public CaptureSettings Capture;
        public SweepSettings Sweep;
        public CoordinateConvention CoordinateConvention;
        public JitterPolicy JitterPolicy;

        public int RandomSeed;
        public string VehicleAssetName;
        public string CameraName;
        public string Notes;

        public string RunRootPath;
        public string ImagesDirectoryPath;
        public string ManifestFilePath;
        public string RunMetadataFilePath;
    }
}
