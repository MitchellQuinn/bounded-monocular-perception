using System;
using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IJitterGenerator
    {
        PoseJitter Generate(RunConfig config, PlannedSample sample, Random rng);
    }
}
