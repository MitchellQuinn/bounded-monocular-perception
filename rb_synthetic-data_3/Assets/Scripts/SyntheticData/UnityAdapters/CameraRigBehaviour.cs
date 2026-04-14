using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.UnityAdapters
{
    public sealed class CameraRigBehaviour : MonoBehaviour, ICameraRig
    {
        [SerializeField] private Camera _camera;

        public Camera GetCamera()
        {
            return _camera;
        }

        public Transform GetCameraTransform()
        {
            return _camera != null ? _camera.transform : null;
        }

        public string GetCameraName()
        {
            return _camera != null ? _camera.name : string.Empty;
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }
        }
#endif
    }
}
