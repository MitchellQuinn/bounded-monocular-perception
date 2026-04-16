using System;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class OutputSettings
    {
        public string OutputRoot = "SyntheticDataset";
        public string ImagesFolderName = "images";
        public string ManifestFolderName = "manifests";
        public string ManifestFileName = "samples.csv";
        public string RunMetadataFileName = "run.json";
    }
}
