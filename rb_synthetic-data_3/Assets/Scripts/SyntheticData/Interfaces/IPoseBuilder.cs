using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IPoseBuilder
    {
        PoseState BuildPose(RunConfig config, PlannedSample sample, PoseJitter jitter);
    }
}
