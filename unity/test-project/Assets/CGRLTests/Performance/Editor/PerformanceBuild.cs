using System;
using System.IO;
using CCNets.CausalGPTRL.Performance;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CCNets.CausalGPTRL.Editor
{
    public static class PerformanceBuild
    {
        public static void BuildWindowsDevelopmentPlayer()
        {
            const string scenePath = "Assets/CGRLTests/Performance/Performance.unity";
            var crawler = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(
                "Assets/CGRLTests/Performance/Models/crawler-b1-ctx32.onnx");
            var soccer = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(
                "Assets/CGRLTests/Performance/Models/soccertwos-b16-ctx32.onnx");
            if (crawler == null || soccer == null)
            {
                throw new InvalidOperationException("Performance ONNX models were not imported as ModelAsset.");
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var gameObject = new GameObject("CGRL Performance Harness");
            var harness = gameObject.AddComponent<PerformanceHarness>();
            harness.models = new[]
            {
                new PerformanceHarness.ModelCase
                {
                    label = "Crawler b1 ctx32",
                    model = crawler,
                    batch = 1,
                    context = 32,
                    stateSize = 158,
                    actionSize = 20,
                },
                new PerformanceHarness.ModelCase
                {
                    label = "SoccerTwos b16 ctx32",
                    model = soccer,
                    batch = 16,
                    context = 32,
                    stateSize = 336,
                    actionSize = 9,
                },
            };
            EditorSceneManager.SaveScene(scene, scenePath);

            var outputDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "builds", "performance"));
            Directory.CreateDirectory(outputDirectory);
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = Path.Combine(outputDirectory, "CGRLPerformance.exe"),
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.Development,
            });
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new InvalidOperationException($"Development Player build failed: {report.summary.result}");
            }
        }
    }
}
