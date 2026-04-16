using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using RaccoonBall.SyntheticData.Config;
using RaccoonBall.SyntheticData.Core;
using RaccoonBall.SyntheticData.Interfaces;
using RaccoonBall.SyntheticData.Runtime;
using UnityEngine;

namespace RaccoonBall.SyntheticData.UnityAdapters
{
    public sealed class RunControllerBehaviour : MonoBehaviour
    {
        private sealed class CellExecutionPlan
        {
            public StratifiedPlacementCell Cell;
            public int TargetSampleCount;
            public float EstimatedAcceptanceRate;
            public float AllocationWeight;
        }

        private sealed class CellProbeStats
        {
            public StratifiedPlacementCell Cell;
            public int ProbeAttempts;
            public int ProbeAccepted;
            public float AcceptanceRate;
        }

        [Header("Scene References")]
        [SerializeField] private RunConfigAsset _runConfigAsset;
        [SerializeField] private VehicleSceneControllerBehaviour _vehicleSceneController;
        [SerializeField] private CameraRigBehaviour _cameraRig;

        [Header("Execution")]
        [SerializeField] private bool _runOnStart;
        [SerializeField] private bool _logPerSample;
        [SerializeField, Min(1)] private int _batchSize = 200;
        [SerializeField] private bool _yieldEveryBatch = true;
        [SerializeField] private bool _flushManifestEveryBatch = true;
        [SerializeField] private bool _runGarbageCollectionEveryBatch;
        [SerializeField] private bool _logBatchDiagnostics = true;
        [SerializeField] private bool _disallowOutputInsideAssets = true;
        [SerializeField, Min(64)] private int _pathLengthWarningThreshold = 180;

        private bool _isRunning;
        private bool _cancelRequested;

        private void Start()
        {
            if (_runOnStart)
            {
                RunGeneration();
            }
        }

        [ContextMenu("Run Generation")]
        public void RunGeneration()
        {
            if (_isRunning)
            {
                Debug.LogWarning("Synthetic data generation is already running.", this);
                return;
            }

            _cancelRequested = false;
            StartCoroutine(RunGenerationCoroutine());
        }

        [ContextMenu("Stop Generation")]
        public void StopGeneration()
        {
            if (!_isRunning)
            {
                Debug.LogWarning("Synthetic data generation is not currently running.", this);
                return;
            }

            _cancelRequested = true;
            Debug.LogWarning("Synthetic data generation stop requested. Run will stop at the next batch boundary.", this);
        }

        private IEnumerator RunGenerationCoroutine()
        {
            _isRunning = true;

            ICaptureService captureService = null;
            IManifestWriter manifestWriter = null;
            RunLogWriter runLogWriter = null;

            try
            {
                ValidateReferences();
                ValidateExecutionSettings();

                RunConfig config = _runConfigAsset.ToRunConfig();
                ValidateConfig(config);
                NormalizeAndValidateOutputPaths(config);
                LogIgnoredLegacySettings(config);

                Camera captureCamera = _cameraRig.GetCamera();
                Transform cameraTransform = _cameraRig.GetCameraTransform();

                captureService = new CaptureService(captureCamera);
                IImageFileWriter imageFileWriter = new ImageFileWriter();
                IFileNamingStrategy fileNamingStrategy = new FileNamingStrategy(_vehicleSceneController.GetVehicleAssetName());
                manifestWriter = new ManifestWriter();
                IRunMetadataWriter runMetadataWriter = new RunMetadataWriter();
                IManifestRowMapper manifestRowMapper = new ManifestRowMapper();
                DistanceCalculator distanceCalculator = new DistanceCalculator();
                StratifiedPlacementPlanner placementPlanner = new StratifiedPlacementPlanner();

                AlignSceneToCoordinateConvention(config);
                EnsureOutputDirectories(config);

                string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
                runLogWriter = new RunLogWriter(runRoot);

                RunMetadata runMetadata = BuildRunMetadata(config, cameraTransform);
                runMetadataWriter.Write(runMetadata, config);
                manifestWriter.Open(config);

                StratifiedPlacementPlan placementPlan = placementPlanner.BuildPlan(config, captureCamera);
                if (placementPlan.TotalSamples <= 0 || placementPlan.Cells.Count == 0)
                {
                    Debug.LogWarning("No placement cells were generated from the current run config.", this);
                    yield break;
                }

                Debug.Log(
                    "Placement plan ready. " +
                    $"cells={placementPlan.Cells.Count} " +
                    $"depth_bands={config.Sweep.DepthBandCount} " +
                    $"lateral_bins={config.Sweep.LateralBinCount} " +
                    $"usable_depth_m=[{placementPlan.UsableDepthMinM:0.###},{placementPlan.UsableDepthMaxM:0.###}] " +
                    $"total_samples={placementPlan.TotalSamples}.",
                    this);

                var projectionValidator = new VehicleProjectionValidator(
                    captureCamera,
                    _vehicleSceneController.GetVehicleRootTransform(),
                    config.Capture,
                    config.Sweep);

                System.Random rng = new System.Random(config.RandomSeed);
                Vector3 baseRotationEulerDeg = config.CoordinateConvention.VehicleBaseRotationEulerDeg;
                List<CellExecutionPlan> cellExecutionPlan = BuildCellExecutionPlan(
                    config,
                    placementPlan,
                    placementPlanner,
                    projectionValidator,
                    baseRotationEulerDeg,
                    rng);
                int plannedTotalSamples = 0;
                for (int i = 0; i < cellExecutionPlan.Count; i++)
                {
                    plannedTotalSamples += cellExecutionPlan[i].TargetSampleCount;
                }

                var totalTimer = System.Diagnostics.Stopwatch.StartNew();
                var batchTimer = System.Diagnostics.Stopwatch.StartNew();
                int processedInBatch = 0;
                int processedTotal = 0;
                bool cancelled = false;

                for (int cellCursor = 0; cellCursor < cellExecutionPlan.Count; cellCursor++)
                {
                    if (_cancelRequested)
                    {
                        cancelled = true;
                        break;
                    }

                    CellExecutionPlan executionCell = cellExecutionPlan[cellCursor];
                    StratifiedPlacementCell cell = executionCell.Cell;
                    int validSamplesInCell = 0;
                    int attemptsInCell = 0;
                    string lastRejectionReason = "n/a";

                    while (validSamplesInCell < executionCell.TargetSampleCount)
                    {
                        if (_cancelRequested)
                        {
                            cancelled = true;
                            break;
                        }

                        if (attemptsInCell >= config.Sweep.MaxAttemptsPerCell)
                        {
                            int shortfall = executionCell.TargetSampleCount - validSamplesInCell;
                            executionCell.TargetSampleCount = validSamplesInCell;

                            if (shortfall > 0)
                            {
                                bool redistributed = TryRedistributeShortfallToRemainingCells(
                                    cellExecutionPlan,
                                    cellCursor + 1,
                                    shortfall);

                                if (!redistributed)
                                {
                                    throw new InvalidOperationException(
                                        "Failed to reach target sample count: no remaining cells available for redistribution. " +
                                        $"cell_index={cell.CellIndex}, depth_band={cell.DepthBandIndex}, lateral_bin={cell.LateralBinIndex}, " +
                                        $"accepted_samples={validSamplesInCell}, shortfall={shortfall}, " +
                                        $"max_attempts_per_cell={config.Sweep.MaxAttemptsPerCell}, last_rejection='{lastRejectionReason}'.");
                                }

                                Debug.LogWarning(
                                    "Placement cell exhausted attempt budget; redistributing remaining quota to later cells. " +
                                    $"cell_index={cell.CellIndex}, depth_band={cell.DepthBandIndex}, lateral_bin={cell.LateralBinIndex}, " +
                                    $"accepted_samples={validSamplesInCell}, redistributed_shortfall={shortfall}, " +
                                    $"max_attempts_per_cell={config.Sweep.MaxAttemptsPerCell}, last_rejection='{lastRejectionReason}'.",
                                    this);
                            }

                            break;
                        }

                        PoseState poseState = null;
                        PoseJitter jitter = null;
                        bool accepted = false;

                        for (int attempt = 0; attempt < config.Sweep.MaxAttemptsPerSample; attempt++)
                        {
                            if (attemptsInCell >= config.Sweep.MaxAttemptsPerCell)
                            {
                                break;
                            }

                            attemptsInCell++;

                            Vector3 candidatePosition = placementPlanner.SamplePositionInCell(placementPlan, cell, rng);
                            PoseJitter candidateJitter = BuildYawOnlyJitter(config, rng);
                            PoseState candidatePose = BuildPoseFromPlacement(candidatePosition, baseRotationEulerDeg, candidateJitter);

                            _vehicleSceneController.ApplyPose(candidatePose.FinalPosition, candidatePose.FinalRotationEulerDeg);

                            if (!projectionValidator.IsPlacementValid(out string rejectionReason))
                            {
                                lastRejectionReason = rejectionReason;
                                continue;
                            }

                            poseState = candidatePose;
                            jitter = candidateJitter;
                            accepted = true;
                            break;
                        }

                        if (!accepted)
                        {
                            continue;
                        }

                        PlannedSample sample = new PlannedSample
                        {
                            FrameIndex = processedTotal,
                            PositionStepIndex = cell.CellIndex,
                            SampleAtPositionIndex = validSamplesInCell,
                            BasePosZM = placementPlanner.ComputeDepthMeters(placementPlan, poseState.BasePosition),
                        };

                        sample.SampleId = fileNamingStrategy.BuildSampleId(sample);

                        CapturedImage capturedImage = captureService.Capture(config.Capture);
                        string imageFilename = fileNamingStrategy.BuildImageFilename(sample);
                        string imageFullPath = fileNamingStrategy.BuildImageFullPath(config, imageFilename);

                        ImageWriteResult imageWriteResult = imageFileWriter.WriteImage(
                            sample.SampleId,
                            imageFilename,
                            imageFullPath,
                            capturedImage);

                        capturedImage = null;

                        if (!imageWriteResult.Success)
                        {
                            string failureMessage = BuildCaptureFailureMessage(sample, jitter, poseState, imageWriteResult);
                            Debug.LogError(failureMessage, this);
                            throw new IOException(failureMessage);
                        }

                        float distanceM = distanceCalculator.CalculateDistanceM(
                            cameraTransform.position,
                            _vehicleSceneController.GetVehicleRootTransform().position);

                        ManifestRow row = manifestRowMapper.Map(
                            config,
                            sample,
                            jitter,
                            poseState,
                            distanceM,
                            imageWriteResult,
                            config.Capture);

                        manifestWriter.AppendRow(row);
                        processedInBatch++;
                        processedTotal++;
                        validSamplesInCell++;

                        if (_logPerSample)
                        {
                            Debug.Log($"Generated {sample.SampleId} -> {imageWriteResult.FullPath}", this);
                        }

                        bool isBatchBoundary =
                            processedInBatch >= _batchSize ||
                            processedTotal == plannedTotalSamples;

                        if (isBatchBoundary)
                        {
                            if (_flushManifestEveryBatch)
                            {
                                manifestWriter.Flush();
                            }

                            if (_runGarbageCollectionEveryBatch)
                            {
                                GC.Collect();
                                GC.WaitForPendingFinalizers();
                            }

                            if (_logBatchDiagnostics)
                            {
                                LogBatchDiagnostics(
                                    processedInBatch,
                                    processedTotal,
                                    plannedTotalSamples,
                                    batchTimer.Elapsed,
                                    totalTimer.Elapsed);
                            }
                            else if (!_logPerSample && (processedTotal % 500 == 0 || processedTotal == plannedTotalSamples))
                            {
                                Debug.Log($"Generated {processedTotal} / {plannedTotalSamples} images.", this);
                            }

                            processedInBatch = 0;
                            batchTimer.Restart();

                            if (_yieldEveryBatch)
                            {
                                yield return null;
                            }
                        }
                    }

                    if (cancelled)
                    {
                        break;
                    }
                }

                if (cancelled || _cancelRequested)
                {
                    Debug.LogWarning("Synthetic data generation stopped before completion.", this);
                }
                else
                {
                    Debug.Log($"Synthetic data generation complete: {processedTotal} images written.", this);
                }
            }
            finally
            {
                try
                {
                    captureService?.Dispose();
                }
                catch (Exception disposeEx)
                {
                    Debug.LogError($"Failed to dispose capture service cleanly. {disposeEx}", this);
                }

                try
                {
                    manifestWriter?.Close();
                }
                catch (Exception closeEx)
                {
                    Debug.LogError($"Failed to close manifest writer cleanly. {closeEx}", this);
                }

                try
                {
                    runLogWriter?.Dispose();
                }
                catch (Exception logCloseEx)
                {
                    Debug.LogError($"Failed to close run log writer cleanly. {logCloseEx}", this);
                }

                _isRunning = false;
                _cancelRequested = false;
            }
        }

        private void ValidateReferences()
        {
            if (_runConfigAsset == null)
                throw new InvalidOperationException("RunConfigAsset reference is missing.");

            if (_vehicleSceneController == null)
                throw new InvalidOperationException("VehicleSceneControllerBehaviour reference is missing.");

            if (_cameraRig == null)
                throw new InvalidOperationException("CameraRigBehaviour reference is missing.");

            if (_cameraRig.GetCamera() == null)
                throw new InvalidOperationException("CameraRigBehaviour has no Camera assigned.");
        }

        private static void ValidateConfig(RunConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (string.IsNullOrWhiteSpace(config.RunId))
                throw new ArgumentException("RunId must not be empty.");

            if (config.Output == null)
                throw new ArgumentException("Output settings must not be null.");

            if (string.IsNullOrWhiteSpace(config.Output.OutputRoot))
                throw new ArgumentException("OutputRoot must not be empty.");

            if (string.IsNullOrWhiteSpace(config.Output.ImagesFolderName))
                throw new ArgumentException("ImagesFolderName must not be empty.");

            if (string.IsNullOrWhiteSpace(config.Output.ManifestFolderName))
                throw new ArgumentException("ManifestFolderName must not be empty.");

            if (string.IsNullOrWhiteSpace(config.Output.ManifestFileName))
                throw new ArgumentException("ManifestFileName must not be empty.");

            if (string.IsNullOrWhiteSpace(config.Output.RunMetadataFileName))
                throw new ArgumentException("RunMetadataFileName must not be empty.");

            if (config.Capture == null)
                throw new ArgumentException("Capture settings must not be null.");

            if (config.Capture.ImageWidthPx <= 0 || config.Capture.ImageHeightPx <= 0)
                throw new ArgumentException("Capture dimensions must be > 0.");

            if (config.Sweep == null)
                throw new ArgumentException("Sweep settings must not be null.");

            if (config.Sweep.DepthBandCount <= 0)
                throw new ArgumentException("DepthBandCount must be > 0.");

            if (config.Sweep.LateralBinCount <= 0)
                throw new ArgumentException("LateralBinCount must be > 0.");

            if (config.Sweep.MaxAttemptsPerSample <= 0)
                throw new ArgumentException("MaxAttemptsPerSample must be > 0.");

            if (config.Sweep.MaxAttemptsPerCell <= 0)
                throw new ArgumentException("MaxAttemptsPerCell must be > 0.");

            if (config.Sweep.FeasibilityProbeAttemptsPerCell <= 0)
                throw new ArgumentException("FeasibilityProbeAttemptsPerCell must be > 0.");

            if (config.Sweep.AcceptanceProbeAttemptsPerCell <= 0)
                throw new ArgumentException("AcceptanceProbeAttemptsPerCell must be > 0.");

            if (config.Sweep.MinSamplesPerFeasibleCell < 0)
                throw new ArgumentException("MinSamplesPerFeasibleCell must be >= 0.");

            if (config.Sweep.MinAcceptanceRateForWeight <= 0f)
                throw new ArgumentException("MinAcceptanceRateForWeight must be > 0.");

            if (config.Sweep.AcceptanceWeightExponent <= 0f)
                throw new ArgumentException("AcceptanceWeightExponent must be > 0.");

            if (config.Sweep.EdgeMarginPx < 0f)
                throw new ArgumentException("EdgeMarginPx must be >= 0.");

            if (config.Sweep.MinProjectedWidthPx < 0f || config.Sweep.MinProjectedHeightPx < 0f || config.Sweep.MinProjectedAreaPx < 0f)
                throw new ArgumentException("Minimum projected size constraints must be >= 0.");

            if (config.Sweep.MaxProjectedWidthPx < 0f || config.Sweep.MaxProjectedHeightPx < 0f || config.Sweep.MaxProjectedAreaPx < 0f)
                throw new ArgumentException("Maximum projected size constraints must be >= 0.");

            if (config.Sweep.MaxProjectedWidthPx > 0f && config.Sweep.MaxProjectedWidthPx < config.Sweep.MinProjectedWidthPx)
                throw new ArgumentException("MaxProjectedWidthPx must be >= MinProjectedWidthPx.");

            if (config.Sweep.MaxProjectedHeightPx > 0f && config.Sweep.MaxProjectedHeightPx < config.Sweep.MinProjectedHeightPx)
                throw new ArgumentException("MaxProjectedHeightPx must be >= MinProjectedHeightPx.");

            if (config.Sweep.MaxProjectedAreaPx > 0f && config.Sweep.MaxProjectedAreaPx < config.Sweep.MinProjectedAreaPx)
                throw new ArgumentException("MaxProjectedAreaPx must be >= MinProjectedAreaPx.");

            if (config.Sweep.UsableDepthMinM > 0f &&
                config.Sweep.UsableDepthMaxM > 0f &&
                config.Sweep.UsableDepthMaxM <= config.Sweep.UsableDepthMinM)
            {
                throw new ArgumentException("UsableDepthMaxM must be > UsableDepthMinM when both are set.");
            }

            _ = StratifiedPlacementPlanner.ResolveTargetSampleCount(config.Sweep);

            if (config.CoordinateConvention == null)
                throw new ArgumentException("CoordinateConvention must not be null.");

            if (config.JitterPolicy == null)
                throw new ArgumentException("JitterPolicy must not be null.");
        }

        private void LogIgnoredLegacySettings(RunConfig config)
        {
            const float epsilon = 0.00001f;

            bool hasPositionJitter =
                Mathf.Abs(config.JitterPolicy.PosXMinM) > epsilon ||
                Mathf.Abs(config.JitterPolicy.PosXMaxM) > epsilon ||
                Mathf.Abs(config.JitterPolicy.PosZMinM) > epsilon ||
                Mathf.Abs(config.JitterPolicy.PosZMaxM) > epsilon;

            if (hasPositionJitter)
            {
                Debug.LogWarning(
                    "Position jitter settings are ignored by stratified placement. " +
                    "Position is sampled from depth-band/lateral-bin cells instead.",
                    this);
            }

            // Pitch and roll jitter settings were removed for this placement model.
        }

        private static PoseJitter BuildYawOnlyJitter(RunConfig config, System.Random rng)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.JitterPolicy == null) throw new ArgumentException("RunConfig.JitterPolicy must not be null.");
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            float yawJitterDeg = NextRange(rng, config.JitterPolicy.RotYMinDeg, config.JitterPolicy.RotYMaxDeg);

            return new PoseJitter
            {
                PosX = 0f,
                PosY = 0f,
                PosZ = 0f,
                RotXDeg = 0f,
                RotYDeg = yawJitterDeg,
                RotZDeg = 0f,
            };
        }

        private static PoseState BuildPoseFromPlacement(
            Vector3 basePosition,
            Vector3 baseRotationEulerDeg,
            PoseJitter jitter)
        {
            if (jitter == null) throw new ArgumentNullException(nameof(jitter));

            return new PoseState
            {
                BasePosition = basePosition,
                BaseRotationEulerDeg = baseRotationEulerDeg,
                FinalPosition = basePosition,
                FinalRotationEulerDeg = new Vector3(
                    baseRotationEulerDeg.x,
                    baseRotationEulerDeg.y + jitter.RotYDeg,
                    baseRotationEulerDeg.z),
            };
        }

        private static float NextRange(System.Random rng, float min, float max)
        {
            if (max < min)
            {
                throw new ArgumentException($"Invalid jitter range: max ({max}) < min ({min}).");
            }

            if (Math.Abs(max - min) < float.Epsilon)
            {
                return min;
            }

            return (float)(min + ((max - min) * rng.NextDouble()));
        }

        private static bool TryRedistributeShortfallToRemainingCells(
            List<CellExecutionPlan> cellExecutionPlan,
            int startIndex,
            int shortfall)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));
            if (shortfall <= 0) return true;
            if (startIndex >= cellExecutionPlan.Count) return false;

            int recipientCount = cellExecutionPlan.Count - startIndex;
            if (recipientCount <= 0)
            {
                return false;
            }

            float weightSum = 0f;
            for (int i = startIndex; i < cellExecutionPlan.Count; i++)
            {
                weightSum += Mathf.Max(cellExecutionPlan[i].AllocationWeight, 0.0001f);
            }

            if (weightSum <= float.Epsilon)
            {
                int evenAdd = shortfall / recipientCount;
                int evenRemainder = shortfall % recipientCount;
                for (int i = 0; i < recipientCount; i++)
                {
                    CellExecutionPlan recipient = cellExecutionPlan[startIndex + i];
                    recipient.TargetSampleCount += evenAdd + (i < evenRemainder ? 1 : 0);
                }

                return true;
            }

            int assigned = 0;
            var fractional = new List<(int index, float frac)>(recipientCount);
            for (int i = startIndex; i < cellExecutionPlan.Count; i++)
            {
                CellExecutionPlan recipient = cellExecutionPlan[i];
                float weight = Mathf.Max(recipient.AllocationWeight, 0.0001f);
                float ideal = shortfall * (weight / weightSum);
                int add = Mathf.FloorToInt(ideal);
                recipient.TargetSampleCount += add;
                assigned += add;
                fractional.Add((i, ideal - add));
            }

            int leftover = shortfall - assigned;
            if (leftover > 0)
            {
                fractional.Sort((a, b) =>
                {
                    int cmp = b.frac.CompareTo(a.frac);
                    if (cmp != 0) return cmp;
                    return a.index.CompareTo(b.index);
                });

                for (int i = 0; i < leftover; i++)
                {
                    cellExecutionPlan[fractional[i].index].TargetSampleCount++;
                }
            }

            return true;
        }

        private List<CellExecutionPlan> BuildCellExecutionPlan(
            RunConfig config,
            StratifiedPlacementPlan placementPlan,
            StratifiedPlacementPlanner placementPlanner,
            VehicleProjectionValidator projectionValidator,
            Vector3 baseRotationEulerDeg,
            System.Random rng)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (placementPlan == null) throw new ArgumentNullException(nameof(placementPlan));
            if (placementPlanner == null) throw new ArgumentNullException(nameof(placementPlanner));
            if (projectionValidator == null) throw new ArgumentNullException(nameof(projectionValidator));
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            int probeAttempts = Mathf.Max(
                config.Sweep.FeasibilityProbeAttemptsPerCell,
                config.Sweep.AcceptanceProbeAttemptsPerCell);

            var probeStats = new List<CellProbeStats>(placementPlan.Cells.Count);

            for (int i = 0; i < placementPlan.Cells.Count; i++)
            {
                StratifiedPlacementCell cell = placementPlan.Cells[i];
                if (cell.TargetSampleCount <= 0)
                {
                    continue;
                }

                CellProbeStats stats = ProbeCellAcceptance(
                    config,
                    placementPlan,
                    placementPlanner,
                    projectionValidator,
                    baseRotationEulerDeg,
                    cell,
                    probeAttempts,
                    rng);

                if (stats.ProbeAccepted > 0)
                {
                    probeStats.Add(stats);
                }
            }

            if (probeStats.Count == 0)
            {
                throw new InvalidOperationException(
                    "No feasible placement cells found for current camera/constraints. " +
                    "Loosen edge margin or projected-size limits, or reduce yaw range.");
            }

            if (probeStats.Count < placementPlan.Cells.Count)
            {
                Debug.LogWarning(
                    "Some placement cells are infeasible under current constraints and were dropped. " +
                    $"feasible_cells={probeStats.Count}/{placementPlan.Cells.Count}",
                    this);
            }

            List<CellExecutionPlan> executionPlan = BuildAcceptanceWeightedExecutionPlan(config, placementPlan.TotalSamples, probeStats);
            if (config.Sweep.LogCellAcceptanceStats)
            {
                LogCellAcceptanceStats(probeStats, executionPlan);
            }

            return executionPlan;
        }

        private CellProbeStats ProbeCellAcceptance(
            RunConfig config,
            StratifiedPlacementPlan placementPlan,
            StratifiedPlacementPlanner placementPlanner,
            VehicleProjectionValidator projectionValidator,
            Vector3 baseRotationEulerDeg,
            StratifiedPlacementCell cell,
            int probeAttempts,
            System.Random rng)
        {
            int accepted = 0;

            for (int i = 0; i < probeAttempts; i++)
            {
                Vector3 candidatePosition = placementPlanner.SamplePositionInCell(placementPlan, cell, rng);
                PoseJitter candidateJitter = BuildYawOnlyJitter(config, rng);
                PoseState candidatePose = BuildPoseFromPlacement(candidatePosition, baseRotationEulerDeg, candidateJitter);

                _vehicleSceneController.ApplyPose(candidatePose.FinalPosition, candidatePose.FinalRotationEulerDeg);
                if (projectionValidator.IsPlacementValid(out _))
                {
                    accepted++;
                }
            }

            return new CellProbeStats
            {
                Cell = cell,
                ProbeAttempts = probeAttempts,
                ProbeAccepted = accepted,
                AcceptanceRate = accepted / (float)probeAttempts,
            };
        }

        private static List<CellExecutionPlan> BuildAcceptanceWeightedExecutionPlan(
            RunConfig config,
            int totalSamples,
            List<CellProbeStats> probeStats)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (probeStats == null) throw new ArgumentNullException(nameof(probeStats));
            if (probeStats.Count == 0) throw new ArgumentException("probeStats must not be empty.");
            if (totalSamples <= 0) throw new ArgumentException("totalSamples must be > 0.");

            int feasibleCount = probeStats.Count;
            int minSamplesPerCell = Mathf.Min(
                config.Sweep.MinSamplesPerFeasibleCell,
                totalSamples / feasibleCount);

            var executionPlan = new List<CellExecutionPlan>(feasibleCount);
            int baselineAssigned = minSamplesPerCell * feasibleCount;
            int remainingSamples = totalSamples - baselineAssigned;

            float minRateFloor = config.Sweep.MinAcceptanceRateForWeight;
            float exponent = config.Sweep.AcceptanceWeightExponent;
            float weightSum = 0f;
            var rawWeights = new float[feasibleCount];

            for (int i = 0; i < feasibleCount; i++)
            {
                CellProbeStats stats = probeStats[i];
                float flooredRate = Mathf.Max(stats.AcceptanceRate, minRateFloor);
                float weight = Mathf.Pow(flooredRate, exponent);
                rawWeights[i] = weight;
                weightSum += weight;

                executionPlan.Add(new CellExecutionPlan
                {
                    Cell = stats.Cell,
                    TargetSampleCount = minSamplesPerCell,
                    EstimatedAcceptanceRate = stats.AcceptanceRate,
                    AllocationWeight = weight,
                });
            }

            if (remainingSamples <= 0)
            {
                return executionPlan;
            }

            if (weightSum <= float.Epsilon)
            {
                int evenAdd = remainingSamples / feasibleCount;
                int evenRemainder = remainingSamples % feasibleCount;
                for (int i = 0; i < feasibleCount; i++)
                {
                    executionPlan[i].TargetSampleCount += evenAdd + (i < evenRemainder ? 1 : 0);
                }

                return executionPlan;
            }

            var fractionalRemainders = new float[feasibleCount];
            int assignedFromWeights = 0;
            for (int i = 0; i < feasibleCount; i++)
            {
                float ideal = remainingSamples * (rawWeights[i] / weightSum);
                int add = Mathf.FloorToInt(ideal);
                executionPlan[i].TargetSampleCount += add;
                fractionalRemainders[i] = ideal - add;
                assignedFromWeights += add;
            }

            int leftover = remainingSamples - assignedFromWeights;
            if (leftover > 0)
            {
                var indices = new List<int>(feasibleCount);
                for (int i = 0; i < feasibleCount; i++)
                {
                    indices.Add(i);
                }

                indices.Sort((a, b) =>
                {
                    int cmp = fractionalRemainders[b].CompareTo(fractionalRemainders[a]);
                    if (cmp != 0) return cmp;
                    return a.CompareTo(b);
                });

                for (int i = 0; i < leftover; i++)
                {
                    executionPlan[indices[i]].TargetSampleCount++;
                }
            }

            return executionPlan;
        }

        private void LogCellAcceptanceStats(
            List<CellProbeStats> probeStats,
            List<CellExecutionPlan> executionPlan)
        {
            if (probeStats == null || executionPlan == null || probeStats.Count == 0 || executionPlan.Count == 0)
            {
                return;
            }

            float minRate = float.MaxValue;
            float maxRate = float.MinValue;
            float sumRate = 0f;

            for (int i = 0; i < probeStats.Count; i++)
            {
                float r = probeStats[i].AcceptanceRate;
                if (r < minRate) minRate = r;
                if (r > maxRate) maxRate = r;
                sumRate += r;
            }

            int minQuota = int.MaxValue;
            int maxQuota = int.MinValue;
            int total = 0;
            for (int i = 0; i < executionPlan.Count; i++)
            {
                int q = executionPlan[i].TargetSampleCount;
                if (q < minQuota) minQuota = q;
                if (q > maxQuota) maxQuota = q;
                total += q;
            }

            Debug.Log(
                "Acceptance-weighted cell quotas ready. " +
                $"feasible_cells={probeStats.Count}, " +
                $"acceptance_rate[min/avg/max]={minRate:0.####}/{(sumRate / probeStats.Count):0.####}/{maxRate:0.####}, " +
                $"quota[min/max]={minQuota}/{maxQuota}, total={total}.",
                this);
        }

        private void ValidateExecutionSettings()
        {
            if (_batchSize <= 0)
            {
                throw new InvalidOperationException("Batch size must be > 0.");
            }

            if (_pathLengthWarningThreshold < 64)
            {
                throw new InvalidOperationException("Path length warning threshold must be >= 64.");
            }
        }

        private void NormalizeAndValidateOutputPaths(RunConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Output == null) throw new ArgumentException("Output settings must not be null.");

            ValidatePathSegment(config.RunId, nameof(config.RunId));
            ValidatePathSegment(config.Output.ImagesFolderName, nameof(config.Output.ImagesFolderName));
            ValidatePathSegment(config.Output.ManifestFolderName, nameof(config.Output.ManifestFolderName));
            ValidateFileName(config.Output.ManifestFileName, nameof(config.Output.ManifestFileName));
            ValidateFileName(config.Output.RunMetadataFileName, nameof(config.Output.RunMetadataFileName));
            if (string.Equals(config.Output.ManifestFileName, config.Output.RunMetadataFileName, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("ManifestFileName and RunMetadataFileName must be different.");
            }

            string normalizedOutputRoot = Path.GetFullPath(config.Output.OutputRoot);
            config.Output.OutputRoot = normalizedOutputRoot;

            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            if (_disallowOutputInsideAssets && IsPathInside(runRoot, Application.dataPath))
            {
                throw new ArgumentException(
                    $"Output path '{runRoot}' resolves inside Unity Assets ('{Application.dataPath}'). " +
                    "For large runs, set OutputRoot outside the project Assets folder.");
            }

            string representativeImagePath = Path.Combine(
                runRoot,
                config.Output.ImagesFolderName,
                "vehicle_f999999_z99.999_j999.png");

            if (representativeImagePath.Length >= _pathLengthWarningThreshold)
            {
                Debug.LogWarning(
                    $"Output path length may be too high for stable high-volume runs. " +
                    $"Representative image path length={representativeImagePath.Length}. " +
                    $"Path='{representativeImagePath}'. Consider using a shorter root path such as D:\\RB_OUT.",
                    this);
            }
        }

        private static bool IsPathInside(string candidatePath, string rootPath)
        {
            string normalizedCandidate = EnsureTrailingSeparator(Path.GetFullPath(candidatePath));
            string normalizedRoot = EnsureTrailingSeparator(Path.GetFullPath(rootPath));
            return normalizedCandidate.StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase);
        }

        private static string EnsureTrailingSeparator(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return path;
            }

            if (!path.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
            {
                path += Path.DirectorySeparatorChar;
            }

            return path;
        }

        private static void ValidatePathSegment(string value, string fieldName)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException($"{fieldName} must not be empty.");
            }

            if (value == "." || value == "..")
            {
                throw new ArgumentException($"{fieldName} must not be '.' or '..'.");
            }

            if (value.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                throw new ArgumentException($"{fieldName} contains invalid characters: '{value}'.");
            }

            if (value.IndexOf(Path.DirectorySeparatorChar) >= 0 || value.IndexOf(Path.AltDirectorySeparatorChar) >= 0)
            {
                throw new ArgumentException($"{fieldName} must be a single path segment, but was '{value}'.");
            }
        }

        private static void ValidateFileName(string value, string fieldName)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException($"{fieldName} must not be empty.");
            }

            if (!string.Equals(value, Path.GetFileName(value), StringComparison.Ordinal))
            {
                throw new ArgumentException($"{fieldName} must be a file name only, but was '{value}'.");
            }

            if (value.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                throw new ArgumentException($"{fieldName} contains invalid characters: '{value}'.");
            }
        }

        private void LogBatchDiagnostics(
            int processedInBatch,
            int processedTotal,
            int plannedTotal,
            TimeSpan batchElapsed,
            TimeSpan totalElapsed)
        {
            long managedBytes = GC.GetTotalMemory(false);
            long workingSetBytes = TryGetWorkingSetBytes();

            string workingSetText = workingSetBytes >= 0
                ? $"{BytesToMegabytes(workingSetBytes):0.0}"
                : "n/a";

            Debug.Log(
                $"Batch complete: +{processedInBatch} images (total {processedTotal}/{plannedTotal}) | " +
                $"batch_time_s={batchElapsed.TotalSeconds:0.00} total_time_s={totalElapsed.TotalSeconds:0.00} | " +
                $"managed_mb={BytesToMegabytes(managedBytes):0.0} working_set_mb={workingSetText}",
                this);
        }

        private static long TryGetWorkingSetBytes()
        {
            try
            {
                using (System.Diagnostics.Process process = System.Diagnostics.Process.GetCurrentProcess())
                {
                    return process.WorkingSet64;
                }
            }
            catch
            {
                return -1;
            }
        }

        private static double BytesToMegabytes(long bytes)
        {
            const double bytesPerMegabyte = 1024d * 1024d;
            return bytes / bytesPerMegabyte;
        }

        private void AlignSceneToCoordinateConvention(RunConfig config)
        {
            _cameraRig.GetCameraTransform().SetPositionAndRotation(
                config.CoordinateConvention.CameraPosition,
                Quaternion.Euler(config.CoordinateConvention.CameraRotationEulerDeg));

            Vector3 vehicleBasePosition = config.CoordinateConvention.VehicleBasePositionAtSweepStart;
            Vector3 vehicleBaseRotation = config.CoordinateConvention.VehicleBaseRotationEulerDeg;

            _vehicleSceneController.ApplyPose(vehicleBasePosition, vehicleBaseRotation);
        }

        private static void EnsureOutputDirectories(RunConfig config)
        {
            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string imagesDirectory = Path.Combine(runRoot, config.Output.ImagesFolderName);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);

            Directory.CreateDirectory(runRoot);
            Directory.CreateDirectory(imagesDirectory);
            Directory.CreateDirectory(manifestDirectory);
        }

        private RunMetadata BuildRunMetadata(RunConfig config, Transform cameraTransform)
        {
            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string imagesDirectory = Path.Combine(runRoot, config.Output.ImagesFolderName);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);
            string manifestPath = Path.Combine(manifestDirectory, config.Output.ManifestFileName);
            string runMetadataPath = Path.Combine(manifestDirectory, config.Output.RunMetadataFileName);

            return new RunMetadata
            {
                RunId = config.RunId,
                DatasetName = config.DatasetName,
                Output = config.Output,
                Capture = config.Capture,
                Sweep = config.Sweep,
                CoordinateConvention = new CoordinateConvention
                {
                    CameraPosition = cameraTransform.position,
                    CameraRotationEulerDeg = cameraTransform.rotation.eulerAngles,
                    VehicleBasePositionAtSweepStart = config.CoordinateConvention.VehicleBasePositionAtSweepStart,
                    VehicleBaseRotationEulerDeg = config.CoordinateConvention.VehicleBaseRotationEulerDeg,
                    SweepAxisName = config.CoordinateConvention.SweepAxisName,
                    WorldUpAxisName = config.CoordinateConvention.WorldUpAxisName,
                    MovingObjectName = config.CoordinateConvention.MovingObjectName,
                    FixedObjectName = config.CoordinateConvention.FixedObjectName,
                    DistanceDefinition = config.CoordinateConvention.DistanceDefinition,
                },
                JitterPolicy = config.JitterPolicy,
                RandomSeed = config.RandomSeed,
                VehicleAssetName = _vehicleSceneController.GetVehicleAssetName(),
                CameraName = _cameraRig.GetCameraName(),
                Notes = config.Notes,
                RunRootPath = runRoot,
                ImagesDirectoryPath = imagesDirectory,
                ManifestFilePath = manifestPath,
                RunMetadataFilePath = runMetadataPath,
            };
        }

        private static string BuildCaptureFailureMessage(
            PlannedSample sample,
            PoseJitter jitter,
            PoseState poseState,
            ImageWriteResult imageWriteResult)
        {
            return
                "Image capture/write failed. " +
                $"sample_id={sample.SampleId}, " +
                $"frame_index={sample.FrameIndex}, " +
                $"base_pos_z_m={sample.BasePosZM:0.######}, " +
                $"jitter_pos=({jitter.PosX:0.######},{jitter.PosY:0.######},{jitter.PosZ:0.######}), " +
                $"jitter_rot=({jitter.RotXDeg:0.######},{jitter.RotYDeg:0.######},{jitter.RotZDeg:0.######}), " +
                $"final_pos=({poseState.FinalPosition.x:0.######},{poseState.FinalPosition.y:0.######},{poseState.FinalPosition.z:0.######}), " +
                $"final_rot=({poseState.FinalRotationEulerDeg.x:0.######},{poseState.FinalRotationEulerDeg.y:0.######},{poseState.FinalRotationEulerDeg.z:0.######}), " +
                $"target_path='{imageWriteResult.FullPath}', " +
                $"error='{imageWriteResult.ErrorMessage}'";
        }
    }
}
