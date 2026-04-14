using System.Collections.Generic;
using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface ISamplePlanner
    {
        IReadOnlyList<PlannedSample> BuildSamples(RunConfig config);
    }
}
