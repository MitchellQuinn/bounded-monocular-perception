using System;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class CaptureService : ICaptureService, IDisposable
    {
        private readonly Camera _camera;
        private RenderTexture _renderTexture;
        private Texture2D _texture;
        private int _currentWidth = -1;
        private int _currentHeight = -1;
        private bool _disposed;

        public CaptureService(Camera camera)
        {
            _camera = camera ? camera : throw new ArgumentNullException(nameof(camera));
        }

        public CapturedImage Capture(CaptureSettings settings)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CaptureService));
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (settings.ImageWidthPx <= 0) throw new ArgumentException("ImageWidthPx must be > 0.");
            if (settings.ImageHeightPx <= 0) throw new ArgumentException("ImageHeightPx must be > 0.");

            EnsureBuffers(settings.ImageWidthPx, settings.ImageHeightPx);

            RenderTexture previousActive = RenderTexture.active;
            RenderTexture previousTargetTexture = _camera.targetTexture;

            try
            {
                _camera.targetTexture = _renderTexture;
                _camera.Render();

                RenderTexture.active = _renderTexture;

                _texture.ReadPixels(new Rect(0, 0, settings.ImageWidthPx, settings.ImageHeightPx), 0, 0, false);
                _texture.Apply(false, false);

                byte[] pngBytes = ImageConversion.EncodeToPNG(_texture);

                return new CapturedImage
                {
                    PngBytes = pngBytes,
                    WidthPx = settings.ImageWidthPx,
                    HeightPx = settings.ImageHeightPx,
                };
            }
            finally
            {
                _camera.targetTexture = previousTargetTexture;
                RenderTexture.active = previousActive;
            }
        }

        private void EnsureBuffers(int width, int height)
        {
            if (_renderTexture != null &&
                _texture != null &&
                _currentWidth == width &&
                _currentHeight == height)
            {
                return;
            }

            ReleaseBuffers();

            _renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            _renderTexture.Create();

            _texture = new Texture2D(width, height, TextureFormat.RGB24, false);

            _currentWidth = width;
            _currentHeight = height;
        }

        private void ReleaseBuffers()
        {
            if (_texture != null)
            {
                UnityEngine.Object.DestroyImmediate(_texture);
                _texture = null;
            }

            if (_renderTexture != null)
            {
                _renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(_renderTexture);
                _renderTexture = null;
            }

            _currentWidth = -1;
            _currentHeight = -1;
        }

        public void Dispose()
        {
            if (_disposed) return;

            ReleaseBuffers();
            _disposed = true;
        }
    }
}