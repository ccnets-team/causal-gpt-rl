using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CCNets.CausalGPTRL;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Profiling;

namespace CCNets.CausalGPTRL.Performance
{
    public sealed class PerformanceHarness : MonoBehaviour
    {
        [Serializable]
        public sealed class ModelCase
        {
            public string label;
            public ModelAsset model;
            public int batch;
            public int context;
            public int stateSize;
            public int actionSize;
        }

        [Serializable]
        private sealed class CaseResult
        {
            public string graph;
            public string backend;
            public int warmupIterations;
            public int measuredIterations;
            public double p50Milliseconds;
            public double p95Milliseconds;
            public long callingThreadAllocatedBytesPerIteration;
            public long maxTotalAllocatedBytesObserved;
            public long maxTotalReservedBytesObserved;
            public string error;
        }

        [Serializable]
        private sealed class Report
        {
            public string unityVersion;
            public string platform;
            public string deviceName;
            public string graphicsDeviceName;
            public string operatingSystem;
            public CaseResult[] cases;
        }

        public ModelCase[] models;
        public int warmupIterations = 10;
        public int measuredIterations = 50;

        private IEnumerator Start()
        {
            Application.runInBackground = true;
            var results = new List<CaseResult>();
            foreach (var modelCase in models)
            {
                results.Add(Measure(modelCase, BackendType.CPU));
                yield return null;
                results.Add(Measure(modelCase, BackendType.GPUCompute));
                yield return null;
            }

            var report = new Report
            {
                unityVersion = Application.unityVersion,
                platform = Application.platform.ToString(),
                deviceName = SystemInfo.deviceName,
                graphicsDeviceName = SystemInfo.graphicsDeviceName,
                operatingSystem = SystemInfo.operatingSystem,
                cases = results.ToArray(),
            };
            var path = ResolveResultsPath();
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            UnityEngine.Debug.Log($"CGRL performance results: {path}");
            Application.Quit(results.Any(result => !string.IsNullOrEmpty(result.error)) ? 1 : 0);
        }

        private CaseResult Measure(ModelCase modelCase, BackendType backendType)
        {
            var result = new CaseResult
            {
                graph = modelCase.label,
                backend = backendType.ToString(),
                warmupIterations = warmupIterations,
                measuredIterations = measuredIterations,
            };
            try
            {
                var statesData = Values(modelCase.batch * modelCase.context * modelCase.stateSize, 0.013f);
                var actionsData = Values(modelCase.batch * modelCase.context * modelCase.actionSize, 0.017f);
                var bosData = new float[modelCase.batch * modelCase.context];
                var maskData = Enumerable.Repeat(1.0f, modelCase.batch * modelCase.context).ToArray();
                for (var row = 0; row < modelCase.batch; row++)
                {
                    bosData[row * modelCase.context] = 1.0f;
                }

                using var states = new Tensor<float>(
                    new TensorShape(modelCase.batch, modelCase.context, modelCase.stateSize), statesData);
                using var actions = new Tensor<float>(
                    new TensorShape(modelCase.batch, modelCase.context, modelCase.actionSize), actionsData);
                using var isBos = new Tensor<float>(
                    new TensorShape(modelCase.batch, modelCase.context, 1), bosData);
                using var mask = new Tensor<float>(
                    new TensorShape(modelCase.batch, modelCase.context), maskData);
                using var backend = new UnityInferenceBackend(modelCase.model, backendType);

                for (var index = 0; index < warmupIterations; index++)
                {
                    backend.Execute(states, actions, isBos, mask);
                }

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                var samples = new double[measuredIterations];
                var maxTotalAllocated = Profiler.GetTotalAllocatedMemoryLong();
                var maxTotalReserved = Profiler.GetTotalReservedMemoryLong();
                var stopwatch = new Stopwatch();
                for (var index = 0; index < measuredIterations; index++)
                {
                    stopwatch.Restart();
                    backend.Execute(states, actions, isBos, mask);
                    stopwatch.Stop();
                    samples[index] = stopwatch.Elapsed.TotalMilliseconds;
                    maxTotalAllocated = Math.Max(maxTotalAllocated, Profiler.GetTotalAllocatedMemoryLong());
                    maxTotalReserved = Math.Max(maxTotalReserved, Profiler.GetTotalReservedMemoryLong());
                }
                var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
                Array.Sort(samples);
                result.p50Milliseconds = Percentile(samples, 0.50);
                result.p95Milliseconds = Percentile(samples, 0.95);
                result.callingThreadAllocatedBytesPerIteration = (allocatedAfter - allocatedBefore) / measuredIterations;
                result.maxTotalAllocatedBytesObserved = maxTotalAllocated;
                result.maxTotalReservedBytesObserved = maxTotalReserved;
            }
            catch (Exception exception)
            {
                result.error = exception.ToString();
            }
            return result;
        }

        private static float[] Values(int count, float scale)
        {
            var values = new float[count];
            for (var index = 0; index < count; index++)
            {
                values[index] = Mathf.Sin(index * scale) * 0.25f;
            }
            return values;
        }

        private static double Percentile(double[] sorted, double percentile)
        {
            var index = Math.Max(0, Math.Min(sorted.Length - 1, (int)Math.Ceiling(percentile * sorted.Length) - 1));
            return sorted[index];
        }

        private static string ResolveResultsPath()
        {
            var arguments = Environment.GetCommandLineArgs();
            for (var index = 0; index < arguments.Length - 1; index++)
            {
                if (arguments[index] == "--cgrl-performance-results")
                {
                    return Path.GetFullPath(arguments[index + 1]);
                }
            }
            return Path.Combine(Application.persistentDataPath, "cgrl-performance-results.json");
        }
    }
}
