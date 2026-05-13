using RaccoonBall.SyntheticData.Core;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace RaccoonBall.SyntheticData.Config
{
    [CreateAssetMenu(menuName = "Raccoon Ball/Synthetic Data/Run Config Asset", fileName = "RunConfigAsset")]
    public sealed class RunConfigAsset : ScriptableObject
    {
        [SerializeField] private RunConfig _config = new RunConfig();

        public RunConfig ToRunConfig()
        {
            string json = JsonUtility.ToJson(_config);
            return JsonUtility.FromJson<RunConfig>(json);
        }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(RunConfigAsset))]
    internal sealed class RunConfigAssetEditor : Editor
    {
        private SerializedProperty _configProperty;

        private void OnEnable()
        {
            _configProperty = serializedObject.FindProperty("_config");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            if (_configProperty == null)
            {
                EditorGUILayout.HelpBox("Serialized field '_config' was not found. Reimport scripts and reopen this asset.", MessageType.Warning);
            }
            else
            {
                EditorGUILayout.PropertyField(_configProperty, includeChildren: true);
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
#endif
}
