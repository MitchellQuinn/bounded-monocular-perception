using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
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

        private sealed class CellRunMetrics
        {
            public CellExecutionPlan Plan;
            public int ExistingSamplesAtStart;
            public int NewSamplesWritten;
            public int FailedPlacementAttempts;
            public int ConsecutiveFailures;
            public int MaxConsecutiveFailures;
            public int RedistributedIn;
            public int RedistributedOut;
            public bool RemovedFromPool;
            public string RemovalReason = string.Empty;
            public string LastRejectionReason = "n/a";

            public int TotalGenerated => ExistingSamplesAtStart + NewSamplesWritten;
        }

        private sealed class ResumeState
        {
            public bool IsResume;
            public bool HasPlacementBinId;
            public int ManifestRowCount;
            public int ImageFileCount;
            public Dictionary<int, int> BinCounts = new Dictionary<int, int>();
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

                ResumeState resumeState = ReadResumeState(config);
                RunMetadata runMetadata = BuildRunMetadata(config, cameraTransform);

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

                int requestedTotalSamples = placementPlan.TotalSamples;
                int validBinCount = cellExecutionPlan.Count;
                ValidateExecutionPlanTotal(cellExecutionPlan, requestedTotalSamples);
                GetExecutionPlanQuotaRange(cellExecutionPlan, out int minCellQuota, out int maxCellQuota);
                Dictionary<int, int> binCounts = BuildInitialBinCounts(cellExecutionPlan);
                ApplyResumeStateToBinCounts(resumeState, binCounts, requestedTotalSamples);
                Dictionary<int, CellRunMetrics> cellMetrics = BuildCellRunMetrics(cellExecutionPlan, binCounts);

                if (resumeState.ManifestRowCount >= requestedTotalSamples)
                {
                    LogFinalCellMetrics(
                        cellMetrics,
                        resumeState.ManifestRowCount,
                        requestedTotalSamples,
                        0,
                        "already_complete");

                    Debug.Log(
                        "Synthetic data generation already complete for this run. " +
                        $"manifest_rows={resumeState.ManifestRowCount}, image_files={resumeState.ImageFileCount}, " +
                        $"target_samples={requestedTotalSamples}.",
                        this);
                    yield break;
                }

                runMetadataWriter.Write(runMetadata, config);
                manifestWriter.Open(config, resumeState.IsResume);

                if (resumeState.IsResume)
                {
                    rng = new System.Random(BuildResumeRandomSeed(config.RandomSeed, resumeState.ManifestRowCount));
                    Debug.Log(
                        "Resuming synthetic data generation from manifest. " +
                        $"existing_samples={resumeState.ManifestRowCount}, image_files={resumeState.ImageFileCount}, " +
                        $"valid_bins={validBinCount}, cell_quota[min/max]={minCellQuota}/{maxCellQuota}, " +
                        $"max_consecutive_failures_per_cell={config.Sweep.MaxConsecutiveFailuresPerCell}, " +
                        $"max_failures_per_cell={config.Sweep.MaxFailuresPerCell}, " +
                        $"remaining_samples={requestedTotalSamples - resumeState.ManifestRowCount}.",
                        this);
                }
                else
                {
                    Debug.Log(
                        "Starting synthetic data generation with resumable bin occupancy. " +
                        $"valid_bins={validBinCount}, cell_quota[min/max]={minCellQuota}/{maxCellQuota}, " +
                        $"max_consecutive_failures_per_cell={config.Sweep.MaxConsecutiveFailuresPerCell}, " +
                        $"max_failures_per_cell={config.Sweep.MaxFailuresPerCell}, target_samples={requestedTotalSamples}.",
                        this);
                }

                List<CellExecutionPlan> eligibleBins = BuildEligibleBins(cellExecutionPlan, binCounts);
                var totalTimer = System.Diagnostics.Stopwatch.StartNew();
                var batchTimer = System.Diagnostics.Stopwatch.StartNew();
                int processedInBatch = 0;
                int processedTotal = resumeState.ManifestRowCount;
                int newSamplesWritten = 0;
                bool cancelled = false;

                while (processedTotal < requestedTotalSamples)
                {
                    if (_cancelRequested)
                    {
                        cancelled = true;
                        break;
                    }

                    if (eligibleBins.Count == 0)
                    {
                        LogFinalCellMetrics(
                            cellMetrics,
                            processedTotal,
                            requestedTotalSamples,
                            eligibleBins.Count,
                            "failed");

                        throw new InvalidOperationException(
                            "Failed to reach target sample count: no eligible placement bins remain after quota redistribution. " +
                            $"generated_total={processedTotal}, target_samples={requestedTotalSamples}, " +
                            $"valid_bins={validBinCount}, " +
                            $"max_consecutive_failures_per_cell={config.Sweep.MaxConsecutiveFailuresPerCell}, " +
                            $"max_failures_per_cell={config.Sweep.MaxFailuresPerCell}.");
                    }

                    int eligibleIndex = rng.Next(eligibleBins.Count);
                    CellExecutionPlan executionCell = eligibleBins[eligibleIndex];
                    StratifiedPlacementCell cell = executionCell.Cell;
                    int placementBinId = GetPlacementBinId(cell);
                    int binSampleCount = binCounts[placementBinId];
                    int binTargetSampleCount = executionCell.TargetSampleCount;
                    CellRunMetrics cellMetricsForBin = cellMetrics[placementBinId];
                    string lastRejectionReason = "n/a";

                    if (binSampleCount >= binTargetSampleCount)
                    {
                        eligibleBins.RemoveAt(eligibleIndex);
                        continue;
                    }

                    if (HasCellExceededFailureBudget(cellMetricsForBin, config.Sweep, out string failureLimitReason))
                    {
                        Debug.LogWarning(
                            "Placement bin exceeded failure budget and its remaining quota was redistributed. " +
                            $"placement_bin_id={placementBinId}, depth_band={cell.DepthBandIndex}, lateral_bin={cell.LateralBinIndex}, " +
                            $"generated_in_bin={binSampleCount}, target_in_bin={binTargetSampleCount}, " +
                            $"failures={cellMetricsForBin.FailedPlacementAttempts}, " +
                            $"consecutive_failures={cellMetricsForBin.ConsecutiveFailures}, " +
                            $"max_consecutive_failures_per_cell={config.Sweep.MaxConsecutiveFailuresPerCell}, " +
                            $"max_failures_per_cell={config.Sweep.MaxFailuresPerCell}, " +
                            $"reason='{failureLimitReason}', " +
                            $"redistributed_shortfall={Mathf.Max(binTargetSampleCount - binSampleCount, 0)}.",
                            this);
                        if (!TryRemoveEligibleBinAndRedistributeShortfall(
                                eligibleBins,
                                eligibleIndex,
                                binCounts,
                                cellMetrics,
                                failureLimitReason))
                        {
                            LogFinalCellMetrics(
                                cellMetrics,
                                processedTotal,
                                requestedTotalSamples,
                                eligibleBins.Count,
                                "failed");

                            throw BuildNoRedistributionTargetFailure(
                                processedTotal,
                                requestedTotalSamples,
                                validBinCount,
                                config.Sweep.MaxConsecutiveFailuresPerCell,
                                config.Sweep.MaxFailuresPerCell);
                        }

                        continue;
                    }

                    PoseState poseState = null;
                    PoseJitter jitter = null;
                    bool accepted = false;

                    for (int attempt = 0; attempt < config.Sweep.MaxAttemptsPerSample; attempt++)
                    {
                        if (HasCellExceededFailureBudget(cellMetricsForBin, config.Sweep, out _))
                        {
                            break;
                        }

                        Vector3 candidatePosition = placementPlanner.SamplePositionInCell(placementPlan, cell, rng);
                        PoseJitter candidateJitter = BuildJitter(config, rng);
                        ApplyCameraJitter(
                            cameraTransform,
                            config.CoordinateConvention.CameraPosition,
                            config.CoordinateConvention.CameraRotationEulerDeg,
                            candidateJitter);
                        PoseState candidatePose = BuildPoseFromPlacement(candidatePosition, baseRotationEulerDeg, candidateJitter);

                        _vehicleSceneController.ApplyPose(candidatePose.FinalPosition, candidatePose.FinalRotationEulerDeg);

                        if (!projectionValidator.IsPlacementValid(out string rejectionReason))
                        {
                            lastRejectionReason = rejectionReason;
                            RecordPlacementFailure(cellMetricsForBin, rejectionReason);
                            continue;
                        }

                        poseState = candidatePose;
                        jitter = candidateJitter;
                        accepted = true;
                        break;
                    }

                    if (!accepted)
                    {
                        if (HasCellExceededFailureBudget(cellMetricsForBin, config.Sweep, out failureLimitReason))
                        {
                            Debug.LogWarning(
                                "Placement bin exceeded failure budget and its remaining quota was redistributed. " +
                                $"placement_bin_id={placementBinId}, depth_band={cell.DepthBandIndex}, lateral_bin={cell.LateralBinIndex}, " +
                                $"generated_in_bin={binSampleCount}, target_in_bin={binTargetSampleCount}, " +
                                $"failures={cellMetricsForBin.FailedPlacementAttempts}, " +
                                $"consecutive_failures={cellMetricsForBin.ConsecutiveFailures}, " +
                                $"max_consecutive_failures_per_cell={config.Sweep.MaxConsecutiveFailuresPerCell}, " +
                                $"max_failures_per_cell={config.Sweep.MaxFailuresPerCell}, " +
                                $"reason='{failureLimitReason}', last_rejection='{lastRejectionReason}'.",
                                this);
                            if (!TryRemoveEligibleBinAndRedistributeShortfall(
                                    eligibleBins,
                                    eligibleIndex,
                                    binCounts,
                                    cellMetrics,
                                    failureLimitReason))
                            {
                                LogFinalCellMetrics(
                                    cellMetrics,
                                    processedTotal,
                                    requestedTotalSamples,
                                    eligibleBins.Count,
                                    "failed");

                                throw BuildNoRedistributionTargetFailure(
                                    processedTotal,
                                    requestedTotalSamples,
                                    validBinCount,
                                    config.Sweep.MaxConsecutiveFailuresPerCell,
                                    config.Sweep.MaxFailuresPerCell);
                            }
                        }

                        continue;
                    }

                    PlannedSample sample = new PlannedSample
                    {
                        FrameIndex = processedTotal,
                        PlacementBinId = placementBinId,
                        PositionStepIndex = cell.CellIndex,
                        SampleAtPositionIndex = binSampleCount,
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
                        string failureMessage = BuildCaptureFailureMessage(sample, jitter, poseState, cameraTransform, imageWriteResult);
                        Debug.LogError(failureMessage, this);
                        LogFinalCellMetrics(
                            cellMetrics,
                            processedTotal,
                            requestedTotalSamples,
                            eligibleBins.Count,
                            "failed");
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
                    newSamplesWritten++;

                    RecordPlacementSuccess(cellMetricsForBin);
                    binSampleCount++;
                    binCounts[placementBinId] = binSampleCount;
                    if (binSampleCount >= executionCell.TargetSampleCount)
                    {
                        eligibleBins.RemoveAt(eligibleIndex);
                    }

                    if (_logPerSample)
                    {
                        Debug.Log($"Generated {sample.SampleId} from placement_bin_id={placementBinId} -> {imageWriteResult.FullPath}", this);
                    }

                    bool isBatchBoundary =
                        processedInBatch >= _batchSize ||
                        processedTotal == requestedTotalSamples;

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
                                requestedTotalSamples,
                                batchTimer.Elapsed,
                                totalTimer.Elapsed);
                        }
                        else if (!_logPerSample && (processedTotal % 500 == 0 || processedTotal == requestedTotalSamples))
                        {
                            Debug.Log($"Generated {processedTotal} / {requestedTotalSamples} images.", this);
                        }

                        processedInBatch = 0;
                        batchTimer.Restart();

                        if (_yieldEveryBatch)
                        {
                            yield return null;
                        }
                    }
                }

                if (cancelled || _cancelRequested)
                {
                    LogFinalCellMetrics(
                        cellMetrics,
                        processedTotal,
                        requestedTotalSamples,
                        eligibleBins.Count,
                        "cancelled");

                    Debug.LogWarning("Synthetic data generation stopped before completion.", this);
                }
                else
                {
                    LogFinalCellMetrics(
                        cellMetrics,
                        processedTotal,
                        requestedTotalSamples,
                        eligibleBins.Count,
                        "completed");

                    Debug.Log(
                        "Synthetic data generation complete. " +
                        $"total_images={processedTotal}, new_images_this_run={newSamplesWritten}.",
                        this);
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

            if (config.Sweep.MaxConsecutiveFailuresPerCell <= 0)
                throw new ArgumentException("MaxConsecutiveFailuresPerCell must be > 0.");

            if (config.Sweep.MaxFailuresPerCell <= 0)
                throw new ArgumentException("MaxFailuresPerCell must be > 0.");

            if (config.Sweep.MaxFailuresPerCell < config.Sweep.MaxConsecutiveFailuresPerCell)
                throw new ArgumentException("MaxFailuresPerCell must be >= MaxConsecutiveFailuresPerCell.");

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

            if (config.CameraJitter == null)
                throw new ArgumentException("CameraJitter must not be null.");

            if (config.VehicleJitter == null)
                throw new ArgumentException("VehicleJitter must not be null.");

            ValidateJitterRange(
                nameof(config.VehicleJitter.RotYMinDeg),
                config.VehicleJitter.RotYMinDeg,
                nameof(config.VehicleJitter.RotYMaxDeg),
                config.VehicleJitter.RotYMaxDeg);

            ValidateJitterRange(
                nameof(config.CameraJitter.PosYMinM),
                config.CameraJitter.PosYMinM,
                nameof(config.CameraJitter.PosYMaxM),
                config.CameraJitter.PosYMaxM);

            ValidateJitterRange(
                nameof(config.CameraJitter.RotXMinDeg),
                config.CameraJitter.RotXMinDeg,
                nameof(config.CameraJitter.RotXMaxDeg),
                config.CameraJitter.RotXMaxDeg);
        }

        private static PoseJitter BuildJitter(RunConfig config, System.Random rng)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.CameraJitter == null) throw new ArgumentException("RunConfig.CameraJitter must not be null.");
            if (config.VehicleJitter == null) throw new ArgumentException("RunConfig.VehicleJitter must not be null.");
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            float yawJitterDeg = NextRange(rng, config.VehicleJitter.RotYMinDeg, config.VehicleJitter.RotYMaxDeg);
            float cameraPosYJitterM = NextRange(rng, config.CameraJitter.PosYMinM, config.CameraJitter.PosYMaxM);
            float cameraRotXJitterDeg = NextRange(rng, config.CameraJitter.RotXMinDeg, config.CameraJitter.RotXMaxDeg);

            return new PoseJitter
            {
                PosX = 0f,
                PosY = 0f,
                PosZ = 0f,
                RotXDeg = 0f,
                RotYDeg = yawJitterDeg,
                RotZDeg = 0f,
                CameraPosYM = cameraPosYJitterM,
                CameraRotXDeg = cameraRotXJitterDeg,
            };
        }

        private static void ApplyCameraJitter(
            Transform cameraTransform,
            Vector3 baseCameraPosition,
            Vector3 baseCameraRotationEulerDeg,
            PoseJitter jitter)
        {
            if (cameraTransform == null) throw new ArgumentNullException(nameof(cameraTransform));
            if (jitter == null) throw new ArgumentNullException(nameof(jitter));

            Vector3 cameraPosition = baseCameraPosition;
            cameraPosition.y += jitter.CameraPosYM;
            Vector3 cameraRotationEulerDeg = baseCameraRotationEulerDeg;
            cameraRotationEulerDeg.x += jitter.CameraRotXDeg;
            cameraTransform.SetPositionAndRotation(cameraPosition, Quaternion.Euler(cameraRotationEulerDeg));
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
                FinalPosition = new Vector3(
                    basePosition.x + jitter.PosX,
                    basePosition.y + jitter.PosY,
                    basePosition.z + jitter.PosZ),
                FinalRotationEulerDeg = new Vector3(
                    baseRotationEulerDeg.x + jitter.RotXDeg,
                    baseRotationEulerDeg.y + jitter.RotYDeg,
                    baseRotationEulerDeg.z + jitter.RotZDeg),
            };
        }

        private static void ValidateJitterRange(string minName, float min, string maxName, float max)
        {
            if (max < min)
            {
                throw new ArgumentException($"{maxName} must be >= {minName}.");
            }
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

        private static void RecordPlacementFailure(CellRunMetrics metrics, string rejectionReason)
        {
            if (metrics == null) throw new ArgumentNullException(nameof(metrics));

            metrics.FailedPlacementAttempts++;
            metrics.ConsecutiveFailures++;
            if (metrics.ConsecutiveFailures > metrics.MaxConsecutiveFailures)
            {
                metrics.MaxConsecutiveFailures = metrics.ConsecutiveFailures;
            }

            metrics.LastRejectionReason = string.IsNullOrWhiteSpace(rejectionReason)
                ? "n/a"
                : rejectionReason;
        }

        private static void RecordPlacementSuccess(CellRunMetrics metrics)
        {
            if (metrics == null) throw new ArgumentNullException(nameof(metrics));

            metrics.NewSamplesWritten++;
            metrics.ConsecutiveFailures = 0;
        }

        private static bool HasCellExceededFailureBudget(
            CellRunMetrics metrics,
            SweepSettings sweep,
            out string reason)
        {
            if (metrics == null) throw new ArgumentNullException(nameof(metrics));
            if (sweep == null) throw new ArgumentNullException(nameof(sweep));

            if (metrics.ConsecutiveFailures >= sweep.MaxConsecutiveFailuresPerCell)
            {
                reason = "max_consecutive_failures";
                return true;
            }

            if (metrics.FailedPlacementAttempts >= sweep.MaxFailuresPerCell)
            {
                reason = "max_failures";
                return true;
            }

            reason = string.Empty;
            return false;
        }

        private static bool TryRemoveEligibleBinAndRedistributeShortfall(
            List<CellExecutionPlan> eligibleBins,
            int eligibleIndex,
            Dictionary<int, int> binCounts,
            Dictionary<int, CellRunMetrics> metricsByPlacementBinId,
            string removalReason)
        {
            if (eligibleBins == null) throw new ArgumentNullException(nameof(eligibleBins));
            if (binCounts == null) throw new ArgumentNullException(nameof(binCounts));
            if (metricsByPlacementBinId == null) throw new ArgumentNullException(nameof(metricsByPlacementBinId));
            if (eligibleIndex < 0 || eligibleIndex >= eligibleBins.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(eligibleIndex));
            }

            CellExecutionPlan exhaustedPlan = eligibleBins[eligibleIndex];
            int placementBinId = GetPlacementBinId(exhaustedPlan.Cell);
            binCounts.TryGetValue(placementBinId, out int generatedCount);
            CellRunMetrics metrics = metricsByPlacementBinId[placementBinId];

            int shortfall = Math.Max(exhaustedPlan.TargetSampleCount - generatedCount, 0);
            if (shortfall > 0)
            {
                exhaustedPlan.TargetSampleCount = generatedCount;
            }

            metrics.RemovedFromPool = true;
            metrics.RemovalReason = string.IsNullOrWhiteSpace(removalReason) ? "removed" : removalReason;
            metrics.RedistributedOut += shortfall;

            eligibleBins.RemoveAt(eligibleIndex);
            return TryRedistributeShortfallToEligibleBins(
                eligibleBins,
                shortfall,
                metricsByPlacementBinId);
        }

        private static bool TryRedistributeShortfallToEligibleBins(
            List<CellExecutionPlan> eligibleBins,
            int shortfall,
            Dictionary<int, CellRunMetrics> metricsByPlacementBinId)
        {
            if (eligibleBins == null) throw new ArgumentNullException(nameof(eligibleBins));
            if (metricsByPlacementBinId == null) throw new ArgumentNullException(nameof(metricsByPlacementBinId));
            if (shortfall <= 0) return true;

            int recipientCount = eligibleBins.Count;
            if (recipientCount <= 0)
            {
                return false;
            }

            float weightSum = 0f;
            for (int i = 0; i < eligibleBins.Count; i++)
            {
                weightSum += Mathf.Max(eligibleBins[i].AllocationWeight, 0.0001f);
            }

            if (weightSum <= float.Epsilon)
            {
                int evenAdd = shortfall / recipientCount;
                int evenRemainder = shortfall % recipientCount;
                for (int i = 0; i < recipientCount; i++)
                {
                    CellExecutionPlan recipient = eligibleBins[i];
                    int add = evenAdd + (i < evenRemainder ? 1 : 0);
                    recipient.TargetSampleCount += add;
                    metricsByPlacementBinId[GetPlacementBinId(recipient.Cell)].RedistributedIn += add;
                }

                return true;
            }

            int assigned = 0;
            var fractional = new List<(int index, float frac)>(recipientCount);
            for (int i = 0; i < eligibleBins.Count; i++)
            {
                CellExecutionPlan recipient = eligibleBins[i];
                float weight = Mathf.Max(recipient.AllocationWeight, 0.0001f);
                float ideal = shortfall * (weight / weightSum);
                int add = Mathf.FloorToInt(ideal);
                recipient.TargetSampleCount += add;
                metricsByPlacementBinId[GetPlacementBinId(recipient.Cell)].RedistributedIn += add;
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
                    CellExecutionPlan recipient = eligibleBins[fractional[i].index];
                    recipient.TargetSampleCount++;
                    metricsByPlacementBinId[GetPlacementBinId(recipient.Cell)].RedistributedIn++;
                }
            }

            return true;
        }

        private static InvalidOperationException BuildNoRedistributionTargetFailure(
            int processedTotal,
            int requestedTotalSamples,
            int validBinCount,
            int maxConsecutiveFailuresPerCell,
            int maxFailuresPerCell)
        {
            return new InvalidOperationException(
                "Failed to reach target sample count: a placement bin exceeded its failure budget, " +
                "but no eligible bins remained to receive its unfilled quota. " +
                $"generated_total={processedTotal}, target_samples={requestedTotalSamples}, " +
                $"valid_bins={validBinCount}, " +
                $"max_consecutive_failures_per_cell={maxConsecutiveFailuresPerCell}, " +
                $"max_failures_per_cell={maxFailuresPerCell}.");
        }

        private static ResumeState ReadResumeState(RunConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            var state = new ResumeState
            {
                ImageFileCount = CountImageFiles(config),
            };

            string manifestPath = GetManifestPath(config);
            if (!File.Exists(manifestPath))
            {
                if (state.ImageFileCount > 0)
                {
                    throw new InvalidOperationException(
                        "Cannot resume synthetic data generation: image files exist but the manifest is missing. " +
                        $"image_files={state.ImageFileCount}, manifest_path='{manifestPath}'.");
                }

                return state;
            }

            using (var reader = new StreamReader(manifestPath, Encoding.UTF8, true))
            {
                string headerLine = reader.ReadLine();
                if (headerLine == null)
                {
                    if (state.ImageFileCount > 0)
                    {
                        throw new InvalidOperationException(
                            "Cannot resume synthetic data generation: image files exist but the manifest is empty. " +
                            $"image_files={state.ImageFileCount}, manifest_path='{manifestPath}'.");
                    }

                    return state;
                }

                List<string> headers;
                try
                {
                    headers = ParseCsvLine(headerLine);
                }
                catch (FormatException ex)
                {
                    throw new InvalidOperationException(
                        "Cannot resume synthetic data generation: malformed manifest CSV header. " +
                        $"manifest_path='{manifestPath}'.",
                        ex);
                }

                var headerIndexByName = new Dictionary<string, int>(StringComparer.Ordinal);
                for (int i = 0; i < headers.Count; i++)
                {
                    if (headerIndexByName.ContainsKey(headers[i]))
                    {
                        throw new InvalidOperationException(
                            $"Cannot resume synthetic data generation: manifest contains duplicate column '{headers[i]}'. " +
                            $"manifest_path='{manifestPath}'.");
                    }

                    headerIndexByName.Add(headers[i], i);
                }

                bool hasPlacementBinId = headerIndexByName.TryGetValue(
                    ManifestRow.PlacementBinIdColumnName,
                    out int placementBinIdColumnIndex);
                state.HasPlacementBinId = hasPlacementBinId;

                string line;
                int lineNumber = 1;
                while ((line = reader.ReadLine()) != null)
                {
                    lineNumber++;
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    state.ManifestRowCount++;

                    if (!hasPlacementBinId)
                    {
                        continue;
                    }

                    List<string> values;
                    try
                    {
                        values = ParseCsvLine(line);
                    }
                    catch (FormatException ex)
                    {
                        throw new InvalidOperationException(
                            $"Cannot resume synthetic data generation: malformed manifest CSV at line {lineNumber}. " +
                            $"manifest_path='{manifestPath}'.",
                        ex);
                    }

                    if (values.Count != headers.Count)
                    {
                        throw new InvalidOperationException(
                            "Cannot resume synthetic data generation: manifest row column count does not match the header. " +
                            $"line={lineNumber}, expected_columns={headers.Count}, actual_columns={values.Count}, " +
                            $"manifest_path='{manifestPath}'.");
                    }

                    if (values.Count <= placementBinIdColumnIndex)
                    {
                        throw new InvalidOperationException(
                            "Cannot resume synthetic data generation: manifest row is missing the placement bin id value. " +
                            $"line={lineNumber}, column='{ManifestRow.PlacementBinIdColumnName}', manifest_path='{manifestPath}'.");
                    }

                    string placementBinText = values[placementBinIdColumnIndex];
                    if (!int.TryParse(placementBinText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int placementBinId) ||
                        placementBinId < 0)
                    {
                        throw new InvalidOperationException(
                            "Cannot resume synthetic data generation: manifest row has an invalid placement bin id. " +
                            $"line={lineNumber}, value='{placementBinText}', manifest_path='{manifestPath}'.");
                    }

                    state.BinCounts.TryGetValue(placementBinId, out int count);
                    state.BinCounts[placementBinId] = count + 1;
                }
            }

            if (state.ManifestRowCount != state.ImageFileCount)
            {
                throw new InvalidOperationException(
                    "Cannot resume synthetic data generation: manifest row count and image file count disagree. " +
                    $"manifest_rows={state.ManifestRowCount}, image_files={state.ImageFileCount}, " +
                    $"manifest_path='{manifestPath}', images_path='{GetImagesDirectory(config)}'.");
            }

            if (state.ManifestRowCount == 0)
            {
                return state;
            }

            state.IsResume = state.HasPlacementBinId;
            return state;
        }

        private static Dictionary<int, int> BuildInitialBinCounts(List<CellExecutionPlan> cellExecutionPlan)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));

            var binCounts = new Dictionary<int, int>(cellExecutionPlan.Count);
            for (int i = 0; i < cellExecutionPlan.Count; i++)
            {
                int placementBinId = GetPlacementBinId(cellExecutionPlan[i].Cell);
                if (binCounts.ContainsKey(placementBinId))
                {
                    throw new InvalidOperationException(
                        $"Placement plan contains duplicate placement bin id {placementBinId}.");
                }

                binCounts.Add(placementBinId, 0);
            }

            return binCounts;
        }

        private static Dictionary<int, CellRunMetrics> BuildCellRunMetrics(
            List<CellExecutionPlan> cellExecutionPlan,
            Dictionary<int, int> binCounts)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));
            if (binCounts == null) throw new ArgumentNullException(nameof(binCounts));

            var metrics = new Dictionary<int, CellRunMetrics>(cellExecutionPlan.Count);
            for (int i = 0; i < cellExecutionPlan.Count; i++)
            {
                CellExecutionPlan plan = cellExecutionPlan[i];
                int placementBinId = GetPlacementBinId(plan.Cell);
                binCounts.TryGetValue(placementBinId, out int existingSamples);
                metrics.Add(
                    placementBinId,
                    new CellRunMetrics
                    {
                        Plan = plan,
                        ExistingSamplesAtStart = existingSamples,
                    });
            }

            return metrics;
        }

        private static void ValidateExecutionPlanTotal(
            List<CellExecutionPlan> cellExecutionPlan,
            int requestedTotalSamples)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));
            if (cellExecutionPlan.Count == 0) throw new ArgumentException("cellExecutionPlan must not be empty.");

            long plannedTotal = 0;
            for (int i = 0; i < cellExecutionPlan.Count; i++)
            {
                int target = cellExecutionPlan[i].TargetSampleCount;
                if (target < 0)
                {
                    throw new InvalidOperationException(
                        $"Placement execution plan has a negative cell quota at index {i}: {target}.");
                }

                plannedTotal += target;
            }

            if (plannedTotal != requestedTotalSamples)
            {
                throw new InvalidOperationException(
                    "Placement execution plan quota total does not match requested sample count. " +
                    $"planned_total={plannedTotal}, target_samples={requestedTotalSamples}.");
            }
        }

        private static void GetExecutionPlanQuotaRange(
            List<CellExecutionPlan> cellExecutionPlan,
            out int minCellQuota,
            out int maxCellQuota)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));
            if (cellExecutionPlan.Count == 0)
            {
                minCellQuota = 0;
                maxCellQuota = 0;
                return;
            }

            minCellQuota = int.MaxValue;
            maxCellQuota = int.MinValue;
            for (int i = 0; i < cellExecutionPlan.Count; i++)
            {
                int target = cellExecutionPlan[i].TargetSampleCount;
                if (target < minCellQuota) minCellQuota = target;
                if (target > maxCellQuota) maxCellQuota = target;
            }
        }

        private static void ApplyResumeStateToBinCounts(
            ResumeState resumeState,
            Dictionary<int, int> binCounts,
            int requestedTotalSamples)
        {
            if (resumeState == null) throw new ArgumentNullException(nameof(resumeState));
            if (binCounts == null) throw new ArgumentNullException(nameof(binCounts));

            if (resumeState.ManifestRowCount > requestedTotalSamples)
            {
                throw new InvalidOperationException(
                    "Cannot resume synthetic data generation: existing manifest already exceeds the configured target. " +
                    $"manifest_rows={resumeState.ManifestRowCount}, target_samples={requestedTotalSamples}.");
            }

            if (resumeState.ManifestRowCount == 0 ||
                resumeState.ManifestRowCount == requestedTotalSamples)
            {
                return;
            }

            if (!resumeState.IsResume)
            {
                throw new InvalidOperationException(
                    "Cannot resume synthetic data generation: existing manifest has samples but no resumable bin column. " +
                    $"required_column='{ManifestRow.PlacementBinIdColumnName}'.");
            }

            foreach (KeyValuePair<int, int> item in resumeState.BinCounts)
            {
                if (!binCounts.ContainsKey(item.Key))
                {
                    throw new InvalidOperationException(
                        "Cannot resume synthetic data generation: manifest references a placement bin that is not feasible " +
                        "under the current scene/configuration. " +
                        $"placement_bin_id={item.Key}, existing_samples_in_bin={item.Value}.");
                }

                binCounts[item.Key] = item.Value;
            }
        }

        private static List<CellExecutionPlan> BuildEligibleBins(
            List<CellExecutionPlan> cellExecutionPlan,
            Dictionary<int, int> binCounts)
        {
            if (cellExecutionPlan == null) throw new ArgumentNullException(nameof(cellExecutionPlan));
            if (binCounts == null) throw new ArgumentNullException(nameof(binCounts));

            var eligibleBins = new List<CellExecutionPlan>(cellExecutionPlan.Count);
            for (int i = 0; i < cellExecutionPlan.Count; i++)
            {
                CellExecutionPlan plan = cellExecutionPlan[i];
                int placementBinId = GetPlacementBinId(plan.Cell);
                if (binCounts.TryGetValue(placementBinId, out int count) && count < plan.TargetSampleCount)
                {
                    eligibleBins.Add(plan);
                }
            }

            return eligibleBins;
        }

        private void LogFinalCellMetrics(
            Dictionary<int, CellRunMetrics> metricsByPlacementBinId,
            int processedTotal,
            int requestedTotalSamples,
            int eligibleBinCount,
            string runStatus)
        {
            if (metricsByPlacementBinId == null || metricsByPlacementBinId.Count == 0)
            {
                return;
            }

            var metrics = new List<CellRunMetrics>(metricsByPlacementBinId.Values);
            metrics.Sort((a, b) => GetPlacementBinId(a.Plan.Cell).CompareTo(GetPlacementBinId(b.Plan.Cell)));

            int completedCells = 0;
            int removedCells = 0;
            int pendingCells = 0;
            int totalExistingSamples = 0;
            int totalNewSamples = 0;
            int totalFailures = 0;
            int totalRedistributedIn = 0;
            int totalRedistributedOut = 0;
            int totalRemaining = 0;
            int totalInitialTarget = 0;
            int totalFinalTarget = 0;
            int maxCellFailures = 0;
            int maxCellFailureBinId = -1;

            for (int i = 0; i < metrics.Count; i++)
            {
                CellRunMetrics item = metrics[i];
                int placementBinId = GetPlacementBinId(item.Plan.Cell);
                int remaining = Math.Max(item.Plan.TargetSampleCount - item.TotalGenerated, 0);

                totalExistingSamples += item.ExistingSamplesAtStart;
                totalNewSamples += item.NewSamplesWritten;
                totalFailures += item.FailedPlacementAttempts;
                totalRedistributedIn += item.RedistributedIn;
                totalRedistributedOut += item.RedistributedOut;
                totalRemaining += remaining;
                totalInitialTarget += item.Plan.TargetSampleCount - item.RedistributedIn + item.RedistributedOut;
                totalFinalTarget += item.Plan.TargetSampleCount;

                if (item.FailedPlacementAttempts > maxCellFailures)
                {
                    maxCellFailures = item.FailedPlacementAttempts;
                    maxCellFailureBinId = placementBinId;
                }

                if (item.RemovedFromPool)
                {
                    removedCells++;
                }
                else if (remaining <= 0)
                {
                    completedCells++;
                }
                else
                {
                    pendingCells++;
                }
            }

            int totalPlacementAttempts = totalNewSamples + totalFailures;
            float runAcceptanceRate = totalPlacementAttempts > 0
                ? totalNewSamples / (float)totalPlacementAttempts
                : 0f;

            Debug.Log(
                "Final cell metrics summary: " +
                $"status={runStatus}, total_images={processedTotal}, target_samples={requestedTotalSamples}, " +
                $"shortfall={Math.Max(requestedTotalSamples - processedTotal, 0)}, " +
                $"cells={metrics.Count}, eligible_remaining={eligibleBinCount}, " +
                $"completed_cells={completedCells}, removed_cells={removedCells}, pending_cells={pendingCells}, " +
                $"existing_samples={totalExistingSamples}, new_samples={totalNewSamples}, " +
                $"placement_failures={totalFailures}, placement_acceptance_rate={runAcceptanceRate:0.####}, " +
                $"quota_initial_total={totalInitialTarget}, quota_final_total={totalFinalTarget}, " +
                $"quota_remaining={totalRemaining}, redistributed_in={totalRedistributedIn}, " +
                $"redistributed_out={totalRedistributedOut}, worst_failure_bin={maxCellFailureBinId}, " +
                $"worst_failure_count={maxCellFailures}.",
                this);

            LogWorstFailureCells(metrics);

            Debug.Log(
                "Final cell metrics columns: placement_bin_id,depth_band,lateral_bin,status,existing_samples,new_samples,total_generated,initial_target,final_target,remaining,failures,max_consecutive_failures,current_consecutive_failures,acceptance_rate,redistributed_in,redistributed_out,removal_reason,last_rejection",
                this);

            for (int i = 0; i < metrics.Count; i++)
            {
                CellRunMetrics item = metrics[i];
                StratifiedPlacementCell cell = item.Plan.Cell;
                int initialTarget = item.Plan.TargetSampleCount - item.RedistributedIn + item.RedistributedOut;
                int remaining = Math.Max(item.Plan.TargetSampleCount - item.TotalGenerated, 0);
                int attempts = item.NewSamplesWritten + item.FailedPlacementAttempts;
                float acceptanceRate = attempts > 0
                    ? item.NewSamplesWritten / (float)attempts
                    : 0f;

                string status = item.RemovedFromPool
                    ? "removed"
                    : remaining <= 0
                        ? "complete"
                        : "pending";

                Debug.Log(
                    "Final cell metrics row: " +
                    $"placement_bin_id={GetPlacementBinId(cell)}, " +
                    $"depth_band={cell.DepthBandIndex}, lateral_bin={cell.LateralBinIndex}, " +
                    $"status={status}, existing_samples={item.ExistingSamplesAtStart}, " +
                    $"new_samples={item.NewSamplesWritten}, total_generated={item.TotalGenerated}, " +
                    $"initial_target={initialTarget}, final_target={item.Plan.TargetSampleCount}, " +
                    $"remaining={remaining}, failures={item.FailedPlacementAttempts}, " +
                    $"max_consecutive_failures={item.MaxConsecutiveFailures}, " +
                    $"current_consecutive_failures={item.ConsecutiveFailures}, " +
                    $"acceptance_rate={acceptanceRate:0.####}, " +
                    $"redistributed_in={item.RedistributedIn}, redistributed_out={item.RedistributedOut}, " +
                    $"removal_reason='{SanitizeLogValue(item.RemovalReason)}', " +
                    $"last_rejection='{SanitizeLogValue(item.LastRejectionReason)}'.",
                    this);
            }
        }

        private void LogWorstFailureCells(List<CellRunMetrics> metrics)
        {
            if (metrics == null || metrics.Count == 0)
            {
                return;
            }

            var worst = new List<CellRunMetrics>(metrics);
            worst.Sort((a, b) =>
            {
                int cmp = b.FailedPlacementAttempts.CompareTo(a.FailedPlacementAttempts);
                if (cmp != 0) return cmp;
                return GetPlacementBinId(a.Plan.Cell).CompareTo(GetPlacementBinId(b.Plan.Cell));
            });

            int count = Math.Min(10, worst.Count);
            var builder = new StringBuilder();
            builder.Append("Final cell metrics worst bins by failures:");
            for (int i = 0; i < count; i++)
            {
                CellRunMetrics item = worst[i];
                int attempts = item.NewSamplesWritten + item.FailedPlacementAttempts;
                float acceptanceRate = attempts > 0
                    ? item.NewSamplesWritten / (float)attempts
                    : 0f;

                builder.Append(' ');
                builder.Append('#');
                builder.Append(i + 1);
                builder.Append("(bin=");
                builder.Append(GetPlacementBinId(item.Plan.Cell).ToString(CultureInfo.InvariantCulture));
                builder.Append(",failures=");
                builder.Append(item.FailedPlacementAttempts.ToString(CultureInfo.InvariantCulture));
                builder.Append(",new=");
                builder.Append(item.NewSamplesWritten.ToString(CultureInfo.InvariantCulture));
                builder.Append(",acceptance=");
                builder.Append(acceptanceRate.ToString("0.####", CultureInfo.InvariantCulture));
                builder.Append(",status=");
                builder.Append(item.RemovedFromPool ? "removed" : "active");
                builder.Append(')');
                if (i < count - 1)
                {
                    builder.Append(',');
                }
            }

            Debug.Log(builder.ToString(), this);
        }

        private static string SanitizeLogValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            return value
                .Replace("\r", " ")
                .Replace("\n", " ")
                .Replace("'", "\"");
        }

        private static int GetPlacementBinId(StratifiedPlacementCell cell)
        {
            if (cell == null) throw new ArgumentNullException(nameof(cell));
            return cell.CellIndex;
        }

        private static int BuildResumeRandomSeed(int randomSeed, int manifestRowCount)
        {
            unchecked
            {
                int seed = randomSeed;
                seed = (seed * 397) ^ manifestRowCount;
                seed = (seed * 397) ^ 0x51f15e5d;
                return seed & 0x7fffffff;
            }
        }

        private static int CountImageFiles(RunConfig config)
        {
            string imagesDirectory = GetImagesDirectory(config);
            if (!Directory.Exists(imagesDirectory))
            {
                return 0;
            }

            return Directory.GetFiles(imagesDirectory, "*.png", SearchOption.TopDirectoryOnly).Length;
        }

        private static string GetImagesDirectory(RunConfig config)
        {
            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            return Path.Combine(runRoot, config.Output.ImagesFolderName);
        }

        private static string GetManifestPath(RunConfig config)
        {
            string runRoot = Path.Combine(config.Output.OutputRoot, config.RunId);
            string manifestDirectory = Path.Combine(runRoot, config.Output.ManifestFolderName);
            return Path.Combine(manifestDirectory, config.Output.ManifestFileName);
        }

        private static List<string> ParseCsvLine(string line)
        {
            var values = new List<string>();
            var value = new StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (inQuotes)
                {
                    if (c == '"')
                    {
                        if (i + 1 < line.Length && line[i + 1] == '"')
                        {
                            value.Append('"');
                            i++;
                        }
                        else
                        {
                            inQuotes = false;
                        }
                    }
                    else
                    {
                        value.Append(c);
                    }

                    continue;
                }

                if (c == ',')
                {
                    values.Add(value.ToString());
                    value.Length = 0;
                }
                else if (c == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    value.Append(c);
                }
            }

            if (inQuotes)
            {
                throw new FormatException("CSV line ended inside a quoted field.");
            }

            values.Add(value.ToString());
            return values;
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
                    "Loosen edge margin or projected-size limits, or reduce vehicle/camera jitter ranges.");
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
                PoseJitter candidateJitter = BuildJitter(config, rng);
                ApplyCameraJitter(
                    _cameraRig.GetCameraTransform(),
                    config.CoordinateConvention.CameraPosition,
                    config.CoordinateConvention.CameraRotationEulerDeg,
                    candidateJitter);
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
                CameraJitter = config.CameraJitter,
                VehicleJitter = config.VehicleJitter,
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
            Transform cameraTransform,
            ImageWriteResult imageWriteResult)
        {
            return
                "Image capture/write failed. " +
                $"sample_id={sample.SampleId}, " +
                $"frame_index={sample.FrameIndex}, " +
                $"base_pos_z_m={sample.BasePosZM:0.######}, " +
                $"jitter_pos=({jitter.PosX:0.######},{jitter.PosY:0.######},{jitter.PosZ:0.######}), " +
                $"jitter_rot=({jitter.RotXDeg:0.######},{jitter.RotYDeg:0.######},{jitter.RotZDeg:0.######}), " +
                $"camera_jitter_pos_y_m={jitter.CameraPosYM:0.######}, " +
                $"camera_jitter_rot_x_deg={jitter.CameraRotXDeg:0.######}, " +
                $"camera_pos=({cameraTransform.position.x:0.######},{cameraTransform.position.y:0.######},{cameraTransform.position.z:0.######}), " +
                $"final_pos=({poseState.FinalPosition.x:0.######},{poseState.FinalPosition.y:0.######},{poseState.FinalPosition.z:0.######}), " +
                $"final_rot=({poseState.FinalRotationEulerDeg.x:0.######},{poseState.FinalRotationEulerDeg.y:0.######},{poseState.FinalRotationEulerDeg.z:0.######}), " +
                $"target_path='{imageWriteResult.FullPath}', " +
                $"error='{imageWriteResult.ErrorMessage}'";
        }
    }
}
