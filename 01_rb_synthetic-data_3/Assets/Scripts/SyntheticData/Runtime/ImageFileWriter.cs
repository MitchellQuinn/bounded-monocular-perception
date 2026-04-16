using System;
using System.Collections.Generic;
using System.IO;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class ImageFileWriter : IImageFileWriter
    {
        private readonly HashSet<string> _ensuredDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        public ImageWriteResult WriteImage(string sampleId, string imageFilename, string fullPath, CapturedImage image)
        {
            if (string.IsNullOrWhiteSpace(sampleId)) throw new ArgumentException("sampleId must not be null or whitespace.");
            if (string.IsNullOrWhiteSpace(imageFilename)) throw new ArgumentException("imageFilename must not be null or whitespace.");
            if (string.IsNullOrWhiteSpace(fullPath)) throw new ArgumentException("fullPath must not be null or whitespace.");
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.PngBytes == null || image.PngBytes.Length == 0) throw new ArgumentException("Captured image contains no PNG bytes.");

            try
            {
                string directory = Path.GetDirectoryName(fullPath);
                if (string.IsNullOrWhiteSpace(directory))
                {
                    throw new IOException($"Could not determine directory for image path '{fullPath}'.");
                }

                EnsureDirectory(directory);
                File.WriteAllBytes(fullPath, image.PngBytes);

                return new ImageWriteResult
                {
                    Success = true,
                    SampleId = sampleId,
                    ImageFilename = imageFilename,
                    FullPath = fullPath,
                    ErrorMessage = string.Empty,
                };
            }
            catch (Exception ex)
            {
                return new ImageWriteResult
                {
                    Success = false,
                    SampleId = sampleId,
                    ImageFilename = imageFilename,
                    FullPath = fullPath,
                    ErrorMessage = ex.ToString(),
                };
            }
        }

        private void EnsureDirectory(string directory)
        {
            if (_ensuredDirectories.Contains(directory))
            {
                return;
            }

            Directory.CreateDirectory(directory);
            _ensuredDirectories.Add(directory);
        }
    }
}
