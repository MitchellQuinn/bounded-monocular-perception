using RaccoonBall.SyntheticData.Core;
using UnityEngine;

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
}
