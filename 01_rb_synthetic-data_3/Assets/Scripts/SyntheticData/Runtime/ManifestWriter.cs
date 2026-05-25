using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Runtime
{
    public sealed class ManifestWriter : IManifestWriter
    {
        private const int BufferSizeBytes = 64 * 1024;
        private const int DefenderKeypointCount = 10;
        private StreamWriter _writer;
        private DefenderKeypointSchemaMetadata _defenderSchemaMetadata;

        public void Open(RunConfig config, bool append)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Output == null) throw new ArgumentException("RunConfig.Output must not be null.");

            TargetSettings targets = config.Targets ?? new TargetSettings();
            if (targets.DefenderAmodalKeypointPose == null)
            {
                throw new ArgumentException(
                    "Synthetic generation requires DefenderAmodalKeypointPose target settings.");
            }
            _defenderSchemaMetadata = targets.DefenderAmodalKeypointPose.Schema;

            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);
            string manifestPath = Path.Combine(manifestDirectory, config.Output.ManifestFileName);
            string expectedHeader = BuildHeader();

            Directory.CreateDirectory(manifestDirectory);
            if (append)
            {
                if (!File.Exists(manifestPath))
                {
                    throw new FileNotFoundException("Cannot append to missing manifest.", manifestPath);
                }

                ValidateExistingHeader(manifestPath, expectedHeader);
                _writer = new StreamWriter(manifestPath, true, Encoding.UTF8, BufferSizeBytes);
            }
            else
            {
                _writer = new StreamWriter(manifestPath, false, Encoding.UTF8, BufferSizeBytes);
                _writer.WriteLine(expectedHeader);
                _writer.Flush();
            }
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

        private static string BuildHeader()
        {
            var columns = new List<string>
            {
                "run_id",
                "sample_id",
                "frame_index",
                "image_filename",
                ManifestRow.PlacementBinIdColumnName,
                "position_step_index",
                "sample_at_position_index",
                "base_pos_x_m",
                "base_pos_y_m",
                "base_pos_z_m",
                "base_rot_x_deg",
                "base_rot_y_deg",
                "base_rot_z_deg",
                "jitter_pos_x_m",
                "jitter_pos_y_m",
                "jitter_pos_z_m",
                "jitter_rot_x_deg",
                "jitter_rot_y_deg",
                "jitter_rot_z_deg",
                "final_pos_x_m",
                "final_pos_y_m",
                "final_pos_z_m",
                "final_rot_x_deg",
                "final_rot_y_deg",
                "final_rot_z_deg",
                "distance_m",
                "image_width_px",
                "image_height_px",
                "capture_success",
                "error_message",
            };

            AddDefenderSchemaMetadataColumns(columns);
            AddDefenderTargetColumns(columns);

            return string.Join(",", columns.ToArray());
        }

        private string ToCsvLine(ManifestRow row)
        {
            var values = new List<string>
            {
                Escape(row.RunId),
                Escape(row.SampleId),
                row.FrameIndex.ToString(CultureInfo.InvariantCulture),
                Escape(row.ImageFilename),
                row.PlacementBinId.ToString(CultureInfo.InvariantCulture),
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
            };

            AddDefenderSchemaMetadataValues(values);
            AddDefenderTargetValues(row, values);

            return string.Join(",", values.ToArray());
        }

        private static void AddDefenderSchemaMetadataColumns(List<string> columns)
        {
            columns.Add("defender_keypoint_schema_version");
            columns.Add("defender_keypoint_schema_hash");
            columns.Add("defender_keypoint_schema_path");
            columns.Add("coordinate_space");
            columns.Add("num_keypoints");
            columns.Add("coordinate_width");
            columns.Add("flattening_order");
        }

        private static void AddDefenderTargetColumns(List<string> columns)
        {
            columns.Add("defender_center_x_m");
            columns.Add("defender_center_y_m");
            columns.Add("defender_center_z_m");

            for (int i = 0; i < DefenderKeypointCount; i++)
            {
                string prefix = $"defender_keypoint_{i:00}";
                columns.Add($"{prefix}_x_m");
                columns.Add($"{prefix}_y_m");
                columns.Add($"{prefix}_z_m");
            }

            for (int i = 0; i < DefenderKeypointCount; i++)
            {
                columns.Add($"defender_keypoint_{i:00}_visible");
            }
        }

        private void AddDefenderSchemaMetadataValues(List<string> values)
        {
            DefenderKeypointSchemaMetadata schema = _defenderSchemaMetadata;
            if (schema == null)
            {
                throw new InvalidOperationException(
                    "Synthetic manifest writer has no Defender keypoint schema metadata configured.");
            }

            values.Add(Escape(schema.SchemaVersion));
            values.Add(Escape(schema.SchemaHash));
            values.Add(Escape(schema.SchemaPath));
            values.Add(Escape(schema.CoordinateSpace));
            values.Add(schema.NumKeypoints.ToString(CultureInfo.InvariantCulture));
            values.Add(schema.CoordinateWidth.ToString(CultureInfo.InvariantCulture));
            values.Add(Escape(schema.FlatteningOrder));
        }

        private static void AddDefenderTargetValues(ManifestRow row, List<string> values)
        {
            DefenderAmodalKeypointPoseTargets targets = row.DefenderAmodalKeypointPoseTargets;
            if (targets == null)
            {
                throw new InvalidOperationException(
                    "Synthetic manifest row has no Defender amodal keypoint pose targets.");
            }

            if (targets.KeypointsCameraSpaceM == null || targets.KeypointsCameraSpaceM.Length != DefenderKeypointCount)
            {
                throw new InvalidOperationException(
                    $"Defender keypoint target array must contain exactly {DefenderKeypointCount} camera-space 3D points.");
            }

            if (targets.KeypointsVisible == null || targets.KeypointsVisible.Length != DefenderKeypointCount)
            {
                throw new InvalidOperationException(
                    $"Defender visibility target array must contain exactly {DefenderKeypointCount} values.");
            }

            values.Add(F(targets.CenterCameraSpaceM.x));
            values.Add(F(targets.CenterCameraSpaceM.y));
            values.Add(F(targets.CenterCameraSpaceM.z));

            for (int i = 0; i < DefenderKeypointCount; i++)
            {
                Vector3 keypoint = targets.KeypointsCameraSpaceM[i];
                values.Add(F(keypoint.x));
                values.Add(F(keypoint.y));
                values.Add(F(keypoint.z));
            }

            for (int i = 0; i < DefenderKeypointCount; i++)
            {
                values.Add(targets.KeypointsVisible[i] ? "1" : "0");
            }
        }

        private static void ValidateExistingHeader(string manifestPath, string expectedHeader)
        {
            using (var reader = new StreamReader(manifestPath, Encoding.UTF8, true))
            {
                string actualHeader = reader.ReadLine();
                if (!string.Equals(actualHeader, expectedHeader, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        "Cannot append to manifest because the existing header does not match the current synthetic label schema. " +
                        $"manifest_path='{manifestPath}'.");
                }
            }
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
