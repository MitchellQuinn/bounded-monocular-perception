using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IFileNamingStrategy
    {
        string BuildSampleId(PlannedSample sample);
        string BuildImageFilename(PlannedSample sample);
        string BuildImageFullPath(RunConfig config, string imageFilename);
    }
}
