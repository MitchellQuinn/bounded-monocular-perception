using System;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Core
{
    public sealed class DefenderAmodalKeypointPoseTargets
    {
        public Vector3 CenterCameraSpaceM;
        public Vector3[] KeypointsCameraSpaceM;
        public bool[] KeypointsVisible;

        public DefenderAmodalKeypointPoseTargets(Vector3 centerCameraSpaceM, Vector3[] keypointsCameraSpaceM, bool[] keypointsVisible)
        {
            CenterCameraSpaceM = centerCameraSpaceM;
            KeypointsCameraSpaceM = keypointsCameraSpaceM ?? throw new ArgumentNullException(nameof(keypointsCameraSpaceM));
            KeypointsVisible = keypointsVisible ?? throw new ArgumentNullException(nameof(keypointsVisible));
        }
    }
}
