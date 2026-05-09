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
            if (config.CameraJitter == null) throw new ArgumentException("RunConfig.CameraJitter must not be null.");
            if (config.VehicleJitter == null) throw new ArgumentException("RunConfig.VehicleJitter must not be null.");
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            var camera = config.CameraJitter;
            var vehicle = config.VehicleJitter;

            return new PoseJitter
            {
                PosX = 0f,
                PosZ = 0f,
                RotYDeg = NextRange(rng, vehicle.RotYMinDeg, vehicle.RotYMaxDeg),
                PosY = 0f,
                RotXDeg = 0f,
                RotZDeg = 0f,
                CameraPosYM = NextRange(rng, camera.PosYMinM, camera.PosYMaxM),
                CameraRotXDeg = NextRange(rng, camera.RotXMinDeg, camera.RotXMaxDeg),
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
