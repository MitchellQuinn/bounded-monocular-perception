using System;
using System.IO;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class RunMetadataWriter : IRunMetadataWriter
    {
        public void Write(RunMetadata metadata, RunConfig config)
        {
            if (metadata == null) throw new ArgumentNullException(nameof(metadata));
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Output == null) throw new ArgumentException("RunConfig.Output must not be null.");

            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);
            string metadataPath = Path.Combine(manifestDirectory, config.Output.RunMetadataFileName);

            Directory.CreateDirectory(manifestDirectory);
            string json = JsonUtility.ToJson(metadata, true);
            File.WriteAllText(metadataPath, json);
        }
    }
}
