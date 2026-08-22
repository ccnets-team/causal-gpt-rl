using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace CCNets.CausalGPTRL.Tests
{
    public sealed class ForwardParityTests
    {
        [Serializable]
        private sealed class TensorFixture
        {
            public string name;
            public int[] shape;
            public float[] data;
        }

        [Serializable]
        private sealed class InputFixture
        {
            public string fixture_version;
            public string onnx_sha256;
            public TensorFixture[] inputs;
        }

        [Serializable]
        private sealed class ExpectedFixture
        {
            public string fixture_version;
            public string reference_backend;
            public TensorFixture output;
            public float cpu_max_abs_tolerance;
            public float gpu_max_abs_tolerance;
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Models))]
        public void ModelAssetImports(string model)
        {
            Assert.That(FixtureModels.LoadModelAsset(model), Is.Not.Null);
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Backends))]
        public void MatchesOnnxRuntime(string model, BackendType backendType)
        {
            var inputsDocument = FixtureModels.LoadJson<InputFixture>(model, "forward-input.json");
            var expectedDocument = FixtureModels.LoadJson<ExpectedFixture>(model, "forward-expected.json");
            Assert.That(inputsDocument.fixture_version, Is.EqualTo("2"));
            Assert.That(expectedDocument.fixture_version, Is.EqualTo("2"));

            using var states = CreateTensor(FindInput(inputsDocument, "states"));
            using var actions = CreateTensor(FindInput(inputsDocument, "actions"));
            using var isBos = CreateTensor(FindInput(inputsDocument, "is_bos"));
            using var mask = CreateTensor(FindInput(inputsDocument, "mask"));
            using var backend = new UnityInferenceBackend(FixtureModels.LoadModelAsset(model), backendType);

            var actual = backend.Execute(states, actions, isBos, mask);
            var expected = expectedDocument.output.data;
            Assert.That(actual.Length, Is.EqualTo(expected.Length));

            var maxAbs = 0.0f;
            var maxIndex = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]), Is.False, $"{backendType} output[{index}] is NaN.");
                Assert.That(float.IsInfinity(actual[index]), Is.False, $"{backendType} output[{index}] is infinite.");
                var error = Math.Abs(actual[index] - expected[index]);
                if (error > maxAbs)
                {
                    maxAbs = error;
                    maxIndex = index;
                }
            }

            var tolerance = backendType == BackendType.CPU
                ? expectedDocument.cpu_max_abs_tolerance
                : expectedDocument.gpu_max_abs_tolerance;
            Debug.Log(
                $"CGRL one-forward parity model={model} backend={backendType} max_abs={maxAbs:R} " +
                $"index={maxIndex} tolerance={tolerance:R}");
            Assert.That(
                maxAbs,
                Is.LessThanOrEqualTo(tolerance),
                $"{model} on {backendType} differs from ONNX Runtime at output index {maxIndex}: " +
                $"actual={actual[maxIndex]:R}, expected={expected[maxIndex]:R}.");
        }

        private static TensorFixture FindInput(InputFixture fixture, string name)
        {
            var value = fixture.inputs.SingleOrDefault(input => input.name == name);
            Assert.That(value, Is.Not.Null, $"Fixture input '{name}' is missing.");
            return value;
        }

        private static Tensor<float> CreateTensor(TensorFixture fixture)
        {
            var shape = new TensorShape(fixture.shape);
            Assert.That(fixture.data.Length, Is.EqualTo(shape.length));
            return new Tensor<float>(shape, fixture.data);
        }
    }
}
