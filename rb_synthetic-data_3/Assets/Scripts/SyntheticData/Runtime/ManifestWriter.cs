using System;
using System.Globalization;
using System.IO;
using System.Text;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class ManifestWriter : IManifestWriter
    {
        private const int BufferSizeBytes = 64 * 1024;
        private StreamWriter _writer;

        public void Open(RunConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Output == null) throw new ArgumentException("RunConfig.Output must not be null.");

            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);
            string manifestPath = Path.Combine(manifestDirectory, config.Output.ManifestFileName);

            Directory.CreateDirectory(manifestDirectory);
            _writer = new StreamWriter(manifestPath, false, Encoding.UTF8, BufferSizeBytes);
            _writer.WriteLine(Header);
            _writer.Flush();
        }

        public void AppendRow(ManifestRow row)
        {
            if (_writer == null) throw new InvalidOperationException("ManifestWriter is not open.");
            if (row == null) throw new ArgumentNullException(nameof(row));

            _writer.WriteLine(ToCsvLine(row));
        }

        public void Flush()
        {
            if (_writer == null) return;
            _writer.Flush();
        }

        public void Close()
        {
            if (_writer == null) return;

            _writer.Flush();
            _writer.Dispose();
            _writer = null;
        }

        private static string Header =>
            "run_id,sample_id,frame_index,image_filename,position_step_index,sample_at_position_index,base_pos_x_m,base_pos_y_m,base_pos_z_m,base_rot_x_deg,base_rot_y_deg,base_rot_z_deg,jitter_pos_x_m,jitter_pos_y_m,jitter_pos_z_m,jitter_rot_x_deg,jitter_rot_y_deg,jitter_rot_z_deg,final_pos_x_m,final_pos_y_m,final_pos_z_m,final_rot_x_deg,final_rot_y_deg,final_rot_z_deg,distance_m,image_width_px,image_height_px,capture_success,error_message";

        private static string ToCsvLine(ManifestRow row)
        {
            return string.Join(",", new[]
            {
                Escape(row.RunId),
                Escape(row.SampleId),
                row.FrameIndex.ToString(CultureInfo.InvariantCulture),
                Escape(row.ImageFilename),
                row.PositionStepIndex.ToString(CultureInfo.InvariantCulture),
                row.SampleAtPositionIndex.ToString(CultureInfo.InvariantCulture),
                F(row.BasePosXM),
                F(row.BasePosYM),
                F(row.BasePosZM),
                F(row.BaseRotXDeg),
                F(row.BaseRotYDeg),
                F(row.BaseRotZDeg),
                F(row.JitterPosXM),
                F(row.JitterPosYM),
                F(row.JitterPosZM),
                F(row.JitterRotXDeg),
                F(row.JitterRotYDeg),
                F(row.JitterRotZDeg),
                F(row.FinalPosXM),
                F(row.FinalPosYM),
                F(row.FinalPosZM),
                F(row.FinalRotXDeg),
                F(row.FinalRotYDeg),
                F(row.FinalRotZDeg),
                F(row.DistanceM),
                row.ImageWidthPx.ToString(CultureInfo.InvariantCulture),
                row.ImageHeightPx.ToString(CultureInfo.InvariantCulture),
                row.CaptureSuccess ? "true" : "false",
                Escape(row.ErrorMessage),
            });
        }

        private static string F(float value)
        {
            return value.ToString("0.######", CultureInfo.InvariantCulture);
        }

        private static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            string escaped = value.Replace("\"", "\"\"");
            return $"\"{escaped}\"";
        }
    }
}
