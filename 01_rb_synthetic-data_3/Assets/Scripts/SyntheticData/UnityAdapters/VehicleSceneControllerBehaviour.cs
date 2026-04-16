using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.UnityAdapters
{
    public sealed class VehicleSceneControllerBehaviour : MonoBehaviour, IVehicleSceneController
    {
        [SerializeField] private string _vehicleAssetName = "Defender90";

        public void ApplyPose(Vector3 position, Vector3 rotationEulerDeg)
        {
            transform.SetPositionAndRotation(position, Quaternion.Euler(rotationEulerDeg));
        }

        public Transform GetVehicleRootTransform()
        {
            return transform;
        }

        public string GetVehicleAssetName()
        {
            return _vehicleAssetName;
        }
    }
}
