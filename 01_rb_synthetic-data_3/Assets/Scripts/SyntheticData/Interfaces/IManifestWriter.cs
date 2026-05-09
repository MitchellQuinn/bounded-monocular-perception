using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface IManifestWriter
    {
        void Open(RunConfig config, bool append);
        void AppendRow(ManifestRow row);
        void Flush();
        void Close();
    }
}
