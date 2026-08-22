using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// A batched scene restarts rows independently — one agent respawning while the other
    /// fifteen keep playing. The failure this guards is row reuse: if a reset row keeps any
    /// of its old trajectory, the next agent to occupy it is served a policy conditioned on
    /// someone else's history, and nothing about the output looks wrong.
    /// </summary>
    public sealed class RowResetParityTests
    {
        [Serializable]
        private sealed class TensorFixture
        {
            public string name;
            public int[] shape;
            public float[] data;
        }

        [Serializable]
        private sealed class RolloutStep
        {
            public int index;
            public int[] reset_rows;
            public TensorFixture[] inputs;
            public TensorFixture raw_output;
            public TensorFixture feedback_action;
            public TensorFixture environment_action;
            public TensorFixture next_state;
            public TensorFixture[] updated_inputs;
        }

        [Serializable]
        private sealed class RowResetFixture
        {
            public string fixture_version;
            public string bos_cache_mode;
            public TensorFixture initial_state;
            public int continuous_size;
            public int[] branches;
            public int env_action_width;
            public float[] action_low;
            public float[] action_high;
            public float cpu_max_abs_tolerance;
            public float gpu_max_abs_tolerance;
            public RolloutStep[] steps;
        }

        [Test]
        public void RowResetFixturesAreStaged()
        {
            // The generator only emits this document for batch > 1, so an all-b1 staging
            // area would leave the parameterized cases empty and the run green.
            Assert.That(
                FixtureModels.WithRowReset(),
                Is.Not.Empty,
                "No staged model carries rowreset-expected.json; a batched fixture is required.");
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.RowResetBackends))]
        public void RowResetMatchesOnnxRuntime(string model, BackendType backendType)
        {
            var fixture = FixtureModels.LoadJson<RowResetFixture>(model, "rowreset-expected.json");
            Assert.That(fixture.fixture_version, Is.EqualTo("2"));
            Assert.That(fixture.bos_cache_mode, Is.EqualTo("discard"));
            Assert.That(
                fixture.steps.Any(step => step.reset_rows != null && step.reset_rows.Length > 0),
                "The fixture resets no row, so it would pass without any reset semantics.");

            var first = fixture.steps[0].inputs;
            var states = Find(first, "states");
            var actions = Find(first, "actions");
            var batch = states.shape[0];
            var context = states.shape[1];
            var stateSize = states.shape[2];
            var actionSize = actions.shape[2];
            var tolerance = backendType == BackendType.CPU
                ? fixture.cpu_max_abs_tolerance
                : fixture.gpu_max_abs_tolerance;

            var layout = new ActionLayout(
                fixture.continuous_size, fixture.branches, fixture.action_low, fixture.action_high);
            var window = new WindowContext(batch, context, stateSize, actionSize);
            window.Reset(fixture.initial_state.data);

            using var backend = new UnityInferenceBackend(FixtureModels.LoadModelAsset(model), backendType);
            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";
                AssertInputs(window.Inputs(), step.inputs, tolerance, $"{label} input");

                var input = window.Inputs();
                using var statesTensor = new Tensor<float>(new TensorShape(batch, context, stateSize), input.States);
                using var actionsTensor = new Tensor<float>(new TensorShape(batch, context, actionSize), input.Actions);
                using var bosTensor = new Tensor<float>(new TensorShape(batch, context, 1), input.IsBos);
                using var maskTensor = new Tensor<float>(new TensorShape(batch, context), input.Mask);

                var raw = backend.Execute(statesTensor, actionsTensor, bosTensor, maskTensor);
                AssertClose(raw, step.raw_output.data, tolerance, $"{label} raw output");

                var decoded = ActionCodec.Decode(raw, batch, layout);
                var feedback = decoded.FeedbackAction;

                // Order matters and is the whole point of this test: the BOS mask is applied
                // for the action just taken, THEN the ended rows are wiped, and only then does
                // the window advance with a per-row episode-start flag.
                window.AfterActDiscardBos();

                var resetRows = step.reset_rows ?? Array.Empty<int>();
                var isBos = new float[batch];
                if (resetRows.Length > 0)
                {
                    window.ResetRows(resetRows);
                    foreach (var row in resetRows)
                    {
                        Array.Clear(feedback, row * actionSize, actionSize);
                        isBos[row] = 1.0f;
                    }
                }

                AssertClose(feedback, step.feedback_action.data, tolerance, $"{label} feedback");
                AssertClose(
                    decoded.EnvironmentAction, step.environment_action.data, tolerance,
                    $"{label} environment action");

                window.Update(step.next_state.data, feedback, isBos);
                AssertInputs(window.Inputs(), step.updated_inputs, tolerance, $"{label} updated window");
            }
        }

        private static void AssertInputs(
            WindowInputs actual,
            TensorFixture[] expected,
            float tolerance,
            string label)
        {
            AssertClose(actual.States, Find(expected, "states").data, tolerance, $"{label} states");
            AssertClose(actual.Actions, Find(expected, "actions").data, tolerance, $"{label} actions");
            AssertClose(actual.IsBos, Find(expected, "is_bos").data, tolerance, $"{label} is_bos");
            AssertClose(actual.Mask, Find(expected, "mask").data, tolerance, $"{label} mask");
        }

        private static void AssertClose(float[] actual, float[] expected, float tolerance, string label)
        {
            Assert.That(actual, Has.Length.EqualTo(expected.Length), label);
            var maxAbs = 0.0f;
            var maxIndex = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]), Is.False, $"{label}[{index}] is NaN.");
                var error = Math.Abs(actual[index] - expected[index]);
                if (error > maxAbs)
                {
                    maxAbs = error;
                    maxIndex = index;
                }
            }
            Assert.That(maxAbs, Is.LessThanOrEqualTo(tolerance), $"{label} max_abs={maxAbs:R} at {maxIndex}");
        }

        private static TensorFixture Find(TensorFixture[] values, string name)
        {
            var value = values.SingleOrDefault(item => item.name == name);
            Assert.That(value, Is.Not.Null, $"Missing tensor fixture '{name}'.");
            return value;
        }
    }
}
