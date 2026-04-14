using System;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class PoseBuilder : IPoseBuilder
    {
        public PoseState BuildPose(RunConfig config, PlannedSample sample, PoseJitter jitter)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.CoordinateConvention == null) throw new ArgumentException("RunConfig.CoordinateConvention must not be null.");
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (jitter == null) throw new ArgumentNullException(nameof(jitter));

            var convention = config.CoordinateConvention;

            var basePosition = new Vector3(
                convention.VehicleBasePositionAtSweepStart.x,
                convention.VehicleBasePositionAtSweepStart.y,
                sample.BasePosZM);

            var baseRotation = convention.VehicleBaseRotationEulerDeg;

            var finalPosition = new Vector3(
                basePosition.x + jitter.PosX,
                basePosition.y + jitter.PosY,
                basePosition.z + jitter.PosZ);

            var finalRotation = new Vector3(
                baseRotation.x + jitter.RotXDeg,
                baseRotation.y + jitter.RotYDeg,
                baseRotation.z + jitter.RotZDeg);

            return new PoseState
            {
                BasePosition = basePosition,
                BaseRotationEulerDeg = baseRotation,
                FinalPosition = finalPosition,
                FinalRotationEulerDeg = finalRotation,
            };
        }
    }
}
