using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IRunMetadataWriter
    {
        void Write(RunMetadata metadata, RunConfig config);
    }
}
