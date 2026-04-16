using System;
using System.Collections.Generic;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class SamplePlanner : ISamplePlanner
    {
        public IReadOnlyList<PlannedSample> BuildSamples(RunConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Sweep == null) throw new ArgumentException("RunConfig.Sweep must not be null.");
            if (config.Sweep.TotalSamples <= 0) throw new ArgumentException("TotalSamples must be > 0.");

            var samples = new List<PlannedSample>();
            for (int frameIndex = 0; frameIndex < config.Sweep.TotalSamples; frameIndex++)
            {
                samples.Add(new PlannedSample
                {
                    FrameIndex = frameIndex,
                    PositionStepIndex = frameIndex,
                    SampleAtPositionIndex = 0,
                    BasePosZM = 0f,
                });
            }

            return samples;
        }
    }
}
