using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// Fixtures are staged one directory per model, so a single run covers several
    /// graphs. Tests enumerate whatever is staged rather than naming a fixed model —
    /// with <see cref="FixturesAreStaged"/> guarding the case that makes an empty
    /// source look like a green run.
    /// </summary>
    internal static class FixtureModels
    {
        internal const string Root = "Assets/CGRLTests/Fixtures";

        /// <summary>The continuous-action model every bundle-level test is written against.</summary>
        internal const string Crawler = "crawler-b1-ctx32";

        /// <summary>
        /// Coverage this suite is supposed to have, named so it cannot quietly shrink. The
        /// parity tests enumerate whatever is staged, so dropping a model from the staging
        /// directory would otherwise leave a smaller run looking just as green: continuous
        /// (b1), a single branch (b16), and several branches (b16).
        /// </summary>
        internal static readonly string[] Required =
        {
            Crawler,
            "pyramids-b16-ctx32",
            "soccertwos-b16-ctx32",
        };

        internal static string[] All()
        {
            return AssetDatabase.GetSubFolders(Root)
                .Select(Path.GetFileName)
                .OrderBy(name => name, System.StringComparer.Ordinal)
                .ToArray();
        }

        /// <summary>Every staged model against every backend, as NUnit cases.</summary>
        internal static IEnumerable<TestCaseData> Backends()
        {
            foreach (var model in All())
            {
                yield return new TestCaseData(model, BackendType.CPU).SetName($"{{m}}_{model}_CPU");
                yield return new TestCaseData(model, BackendType.GPUCompute).SetName($"{{m}}_{model}_GPUCompute");
            }
        }

        /// <summary>
        /// The same cases for a <c>[UnityTest]</c>. Separate from <see cref="Backends"/> only
        /// because of NUnit plumbing: a coroutine test returns an IEnumerator, and a case that
        /// declares no expected result rejects it as "non-void return value, but no result is
        /// expected". Returns(null) declares one.
        /// </summary>
        internal static IEnumerable<TestCaseData> CoroutineBackends()
        {
            foreach (var data in Backends())
            {
                yield return data.Returns(null);
            }
        }

        internal static IEnumerable<TestCaseData> Models()
        {
            foreach (var model in All())
            {
                yield return new TestCaseData(model).SetName($"{{m}}_{model}");
            }
        }

        /// <summary>Staged models carrying a row-reset fixture (the generator emits it for batch > 1).</summary>
        internal static string[] WithRowReset()
        {
            return All()
                .Where(model => AssetDatabase.LoadAssetAtPath<TextAsset>(
                    PathOf(model, "rowreset-expected.json")) != null)
                .ToArray();
        }

        internal static IEnumerable<TestCaseData> RowResetBackends()
        {
            foreach (var model in WithRowReset())
            {
                yield return new TestCaseData(model, BackendType.CPU).SetName($"{{m}}_{model}_CPU");
                yield return new TestCaseData(model, BackendType.GPUCompute).SetName($"{{m}}_{model}_GPUCompute");
            }
        }

        internal static IEnumerable<TestCaseData> RequiredModels()
        {
            foreach (var model in Required)
            {
                yield return new TestCaseData(model).SetName($"{{m}}_{model}");
            }
        }

        internal static string PathOf(string model, string fileName)
        {
            return $"{Root}/{model}/{fileName}";
        }

        internal static ModelAsset LoadModelAsset(string model)
        {
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(PathOf(model, "policy.onnx"));
            Assert.That(asset, Is.Not.Null, $"Unity did not import {model}/policy.onnx as a ModelAsset.");
            return asset;
        }

        internal static string LoadText(string model, string fileName)
        {
            var asset = AssetDatabase.LoadAssetAtPath<TextAsset>(PathOf(model, fileName));
            Assert.That(asset, Is.Not.Null, $"Missing staged fixture {model}/{fileName}.");
            return asset.text;
        }

        internal static T LoadJson<T>(string model, string fileName)
        {
            var value = JsonUtility.FromJson<T>(LoadText(model, fileName));
            Assert.That(value, Is.Not.Null, $"Could not parse staged fixture {model}/{fileName}.");
            return value;
        }
    }

    public sealed class FixtureStagingTests
    {
        [Test]
        public void FixturesAreStaged()
        {
            // Without this, an empty staging directory turns every parameterized parity
            // test into zero cases, and the run reports green having verified nothing.
            Assert.That(
                FixtureModels.All(),
                Is.Not.Empty,
                $"No model directories staged under {FixtureModels.Root}. Run tools/stage_fixtures.ps1.");
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.RequiredModels))]
        public void RequiredModelIsStaged(string model)
        {
            Assert.That(
                FixtureModels.All(),
                Contains.Item(model),
                $"{model} is not staged, so nothing in this run covers what it was there to cover.");
        }
    }
}
