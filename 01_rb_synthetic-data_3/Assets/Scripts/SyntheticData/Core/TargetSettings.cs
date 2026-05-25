using System;
using UnityEngine;

namespace RaccoonBall.SyntheticData.Core
{
    [Serializable]
    public sealed class TargetSettings
    {
        public DefenderAmodalKeypointPoseTargetSettings DefenderAmodalKeypointPose =
            new DefenderAmodalKeypointPoseTargetSettings();
    }

    [Serializable]
    public sealed class DefenderKeypointSchemaMetadata
    {
        public string SchemaVersion = "0.1.0";
        public string SchemaId = "defender_amodal_keypoint_schema";
        public string SchemaStatus = "active";
        public bool TrainingAllowed = true;
        public string SchemaHash = "157d0224ec463338b79855fb26c58e1b01242c7d83c016c122ba2a3135e4eb93";
        public string SchemaPath = "03_rb-training-v2.0/schemas/defender_keypoint_schema.json";
        public string CoordinateSpace = "camera_space_3d";
        public int NumKeypoints = 10;
        public int CoordinateWidth = 3;
        public string FlatteningOrder = "keypoint_major_xyz";
    }

    [Serializable]
    public sealed class DefenderAmodalKeypointDefinition
    {
        public int Index;
        public string Name;
        public string Status;

        public DefenderAmodalKeypointDefinition()
        {
        }

        public DefenderAmodalKeypointDefinition(int index, string name, string status)
        {
            Index = index;
            Name = name;
            Status = status;
        }
    }

    [Serializable]
    public sealed class DefenderAmodalKeypointPoseTargetSettings
    {
        public DefenderKeypointSchemaMetadata Schema = new DefenderKeypointSchemaMetadata();
        public DefenderAmodalKeypointDefinition[] Keypoints = CreateDefaultKeypoints();
        public string RightAxisName = "X";
        public string UpAxisName = "Y";
        public string ForwardAxisName = "Z";
        public int ForwardAxisSign = 1;
        public float BodyHeightFractionOfBounds = 0.55f;
        public float ScreenTopForwardFractionFromRear = 0.62f;
        public bool RequireVisibilityColliders;
        public LayerMask VisibilityLayerMask = ~0;
        public float VisibilitySurfaceToleranceM = 0.03f;

        public void EnsureDefaults()
        {
            if (Schema == null)
            {
                Schema = new DefenderKeypointSchemaMetadata();
            }

            if (Keypoints == null || Keypoints.Length != 10)
            {
                Keypoints = CreateDefaultKeypoints();
            }

            for (int i = 0; i < Keypoints.Length; i++)
            {
                if (Keypoints[i] == null)
                {
                    Keypoints[i] = CreateDefaultKeypoints()[i];
                }
            }
        }

        public static DefenderAmodalKeypointDefinition[] CreateDefaultKeypoints()
        {
            return new[]
            {
                new DefenderAmodalKeypointDefinition(0, "rear_bottom_left", "defined"),
                new DefenderAmodalKeypointDefinition(1, "rear_bottom_right", "defined"),
                new DefenderAmodalKeypointDefinition(2, "rear_roof_left", "defined"),
                new DefenderAmodalKeypointDefinition(3, "rear_roof_right", "defined"),
                new DefenderAmodalKeypointDefinition(4, "front_bottom_left", "defined"),
                new DefenderAmodalKeypointDefinition(5, "front_bottom_right", "defined"),
                new DefenderAmodalKeypointDefinition(6, "front_body_left", "defined"),
                new DefenderAmodalKeypointDefinition(7, "front_body_right", "defined"),
                new DefenderAmodalKeypointDefinition(8, "screen_top_left", "defined"),
                new DefenderAmodalKeypointDefinition(9, "screen_top_right", "defined"),
            };
        }
    }
}
