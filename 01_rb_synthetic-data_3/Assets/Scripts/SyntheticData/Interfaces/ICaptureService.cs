using System;
using RaccoonBall.SyntheticData.Core;

namespace RaccoonBall.SyntheticData.Interfaces
{
    public interface ICaptureService : IDisposable
    {
        CapturedImage Capture(CaptureSettings settings);
    }
}