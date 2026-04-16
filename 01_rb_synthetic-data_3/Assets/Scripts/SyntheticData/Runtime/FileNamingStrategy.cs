using System;
using System.Globalization;
using System.IO;
using System.Text;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class FileNamingStrategy : IFileNamingStrategy
    {
        private readonly string _vehicleSlug;

        public FileNamingStrategy(string vehicleAssetName)
        {
            _vehicleSlug = Sanitize(vehicleAssetName);
            if (string.IsNullOrWhiteSpace(_vehicleSlug))
            {
                _vehicleSlug = "vehicle";
            }
        }

        public string BuildSampleId(PlannedSample sample)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));

            return string.Format(
                CultureInfo.InvariantCulture,
                "{0}_f{1:D6}_z{2:00.000}_j{3:D3}",
                _vehicleSlug,
                sample.FrameIndex,
                sample.BasePosZM,
                sample.SampleAtPositionIndex);
        }

        public string BuildImageFilename(PlannedSample sample)
        {
            return BuildSampleId(sample) + ".png";
        }

        public string BuildImageFullPath(RunConfig config, string imageFilename)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Output == null) throw new ArgumentException("RunConfig.Output must not be null.");
            if (string.IsNullOrWhiteSpace(imageFilename)) throw new ArgumentException("imageFilename must not be null or whitespace.");

            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string imagesDirectory = Path.Combine(runRoot, config.Output.ImagesFolderName);
            return Path.Combine(imagesDirectory, imageFilename);
        }

        private static string Sanitize(string input)
        {
            if (string.IsNullOrWhiteSpace(input)) return string.Empty;

            var sb = new StringBuilder(input.Length);
            foreach (char c in input)
            {
                if (char.IsLetterOrDigit(c))
                {
                    sb.Append(char.ToLowerInvariant(c));
                }
                else if (c == '-' || c == '_')
                {
                    sb.Append(c);
                }
            }

            return sb.ToString();
        }
    }
}
