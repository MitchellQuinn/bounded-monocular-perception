using UnityEngine;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IVehicleSceneController
    {
        void ApplyPose(Vector3 position, Vector3 rotationEulerDeg);
        Transform GetVehicleRootTransform();
        string GetVehicleAssetName();
    }
}
