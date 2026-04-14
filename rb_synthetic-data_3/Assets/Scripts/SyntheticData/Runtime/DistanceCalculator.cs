using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class DistanceCalculator
    {
        public float CalculateDistanceM(Vector3 cameraPosition, Vector3 vehiclePosition)
        {
            return Vector3.Distance(cameraPosition, vehiclePosition);
        }
    }
}
