using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class RunLogWriter : IDisposable
    {
        private readonly object _syncRoot = new object();
        private readonly StreamWriter _writer;
        private bool _disposed;

        public string FilePath { get; }

        public RunLogWriter(string runRootPath)
        {
            if (string.IsNullOrWhiteSpace(runRootPath))
                throw new ArgumentException("runRootPath must not be null or whitespace.", nameof(runRootPath));

            FilePath = Path.Combine(runRootPath, "runlog.txt");
            string directoryPath = Path.GetDirectoryName(FilePath);
            if (!string.IsNullOrWhiteSpace(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            _writer = new StreamWriter(
                new FileStream(FilePath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite),
                new UTF8Encoding(false))
            {
                AutoFlush = true,
            };

            Application.logMessageReceivedThreaded += HandleLogMessageReceived;
        }

        public void Dispose()
        {
            lock (_syncRoot)
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;
            }

            Application.logMessageReceivedThreaded -= HandleLogMessageReceived;

            lock (_syncRoot)
            {
                _writer.Dispose();
            }
        }

        private void HandleLogMessageReceived(string condition, string stackTrace, LogType _)
        {
            lock (_syncRoot)
            {
                if (_disposed)
                {
                    return;
                }

                try
                {
                    _writer.WriteLine(condition);
                    if (!string.IsNullOrWhiteSpace(stackTrace))
                    {
                        _writer.WriteLine(stackTrace);
                    }
                }
                catch
                {
                    // Avoid recursive logging if writing to the run log fails.
                }
            }
        }
    }
}
