using System;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class ManifestRowMapper : IManifestRowMapper
    {
        public ManifestRow Map(
            RunConfig config,
            PlannedSample sample,
            PoseJitter jitter,
            PoseState poseState,
            float distanceM,
            ImageWriteResult imageWriteResult,
            CaptureSettings captureSettings)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (jitter == null) throw new ArgumentNullException(nameof(jitter));
            if (poseState == null) throw new ArgumentNullException(nameof(poseState));
            if (imageWriteResult == null) throw new ArgumentNullException(nameof(imageWriteResult));
            if (captureSettings == null) throw new ArgumentNullException(nameof(captureSettings));

            return new ManifestRow
            {
                RunId = config.RunId,
                SampleId = sample.SampleId,
                FrameIndex = sample.FrameIndex,
                ImageFilename = imageWriteResult.ImageFilename,

                PlacementBinId = sample.PlacementBinId,
                PositionStepIndex = sample.PositionStepIndex,
                SampleAtPositionIndex = sample.SampleAtPositionIndex,

                BasePosXM = poseState.BasePosition.x,
                BasePosYM = poseState.BasePosition.y,
                BasePosZM = poseState.BasePosition.z,

                BaseRotXDeg = poseState.BaseRotationEulerDeg.x,
                BaseRotYDeg = poseState.BaseRotationEulerDeg.y,
                BaseRotZDeg = poseState.BaseRotationEulerDeg.z,

                JitterPosXM = jitter.PosX,
                JitterPosYM = jitter.PosY,
                JitterPosZM = jitter.PosZ,

                JitterRotXDeg = jitter.RotXDeg,
                JitterRotYDeg = jitter.RotYDeg,
                JitterRotZDeg = jitter.RotZDeg,

                FinalPosXM = poseState.FinalPosition.x,
                FinalPosYM = poseState.FinalPosition.y,
                FinalPosZM = poseState.FinalPosition.z,

                FinalRotXDeg = poseState.FinalRotationEulerDeg.x,
                FinalRotYDeg = poseState.FinalRotationEulerDeg.y,
                FinalRotZDeg = poseState.FinalRotationEulerDeg.z,

                DistanceM = distanceM,

                ImageWidthPx = captureSettings.ImageWidthPx,
                ImageHeightPx = captureSettings.ImageHeightPx,

                CaptureSuccess = imageWriteResult.Success,
                ErrorMessage = imageWriteResult.ErrorMessage ?? string.Empty,
            };
        }
    }
}
