using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IImageFileWriter
    {
        ImageWriteResult WriteImage(string sampleId, string imageFilename, string fullPath, CapturedImage image);
    }
}
