using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IManifestRowMapper
    {
        ManifestRow Map(
            RunConfig config,
            PlannedSample sample,
            PoseJitter jitter,
            PoseState poseState,
            float distanceM,
            ImageWriteResult imageWriteResult,
            CaptureSettings captureSettings);
    }
}
