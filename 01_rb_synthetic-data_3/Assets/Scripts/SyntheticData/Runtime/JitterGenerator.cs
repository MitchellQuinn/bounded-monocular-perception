using System;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class JitterGenerator : IJitterGenerator
    {
        public PoseJitter Generate(RunConfig config, PlannedSample sample, Random rng)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.JitterPolicy == null) throw new ArgumentException("RunConfig.JitterPolicy must not be null.");
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            var p = config.JitterPolicy;

            return new PoseJitter
            {
                PosX = NextRange(rng, p.PosXMinM, p.PosXMaxM),
                PosZ = NextRange(rng, p.PosZMinM, p.PosZMaxM),
                RotYDeg = NextRange(rng, p.RotYMinDeg, p.RotYMaxDeg),
                PosY = 0f,
                RotXDeg = 0f,
                RotZDeg = 0f,
            };
        }

        private static float NextRange(Random rng, float min, float max)
        {
            if (max < min)
            {
                throw new ArgumentException($"Invalid jitter range: max ({max}) < min ({min}).");
            }

            if (Math.Abs(max - min) < float.Epsilon)
            {
                return min;
            }

            double t = rng.NextDouble();
            return (float)(min + ((max - min) * t));
        }
    }
}
