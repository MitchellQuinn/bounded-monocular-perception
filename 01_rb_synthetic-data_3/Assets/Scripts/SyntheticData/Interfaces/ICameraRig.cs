using UnityEngine;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface ICameraRig
    {
        Camera GetCamera();
        Transform GetCameraTransform();
        string GetCameraName();
    }
}
