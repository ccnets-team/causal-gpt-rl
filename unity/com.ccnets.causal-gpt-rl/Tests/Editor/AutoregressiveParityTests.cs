using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace CCNets.CausalGPTRL.Tests
{
    public sealed class AutoregressiveParityTests
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
            public TensorFixture[] inputs;
            public TensorFixture raw_output;
            public TensorFixture feedback_action;
            public TensorFixture environment_action;
            public TensorFixture next_state;
            public TensorFixture[] updated_inputs;
        }

        [Serializable]
        private sealed class RolloutFixture
        {
            public string fixture_version;
            public string reference_backend;
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

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Models))]
        public void LayoutFromConfigMatchesFixture(string model)
        {
            // The fixture's layout came from the Python generator; this one comes from the
            // bundle config through C#. They have to agree, or the runtime would decode a
            // different action split than the recorded reference did.
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            var layout = ActionLayout.FromConfig(
                BundleConfig.FromJson(FixtureModels.LoadText(model, "config.json")));

            Assert.That(layout.ContinuousSize, Is.EqualTo(fixture.continuous_size));
            Assert.That(layout.BranchSizes, Is.EqualTo(fixture.branches));
            Assert.That(layout.EnvironmentActionSize, Is.EqualTo(fixture.env_action_width));

            // The bounds are what the decode clips against, and both sides now read them
            // from the bundle instead of assuming [-1, 1]. That makes them the one part of
            // the layout that can differ between the generator and this runtime, so the
            // test that exists to compare the two has to compare them.
            Assert.That(layout.Low, Is.EqualTo(fixture.action_low).Within(1e-6f),
                "The generator clipped the recorded environment action to different lower bounds.");
            Assert.That(layout.High, Is.EqualTo(fixture.action_high).Within(1e-6f),
                "The generator clipped the recorded environment action to different upper bounds.");
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Backends))]
        public void MultiStepRolloutMatchesOnnxRuntime(string model, BackendType backendType)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            Assert.That(fixture.fixture_version, Is.EqualTo("2"));
            Assert.That(fixture.bos_cache_mode, Is.EqualTo("discard"));
            Assert.That(fixture.steps, Has.Length.GreaterThanOrEqualTo(10));

            var firstInputs = fixture.steps[0].inputs;
            var states = Find(firstInputs, "states");
            var actions = Find(firstInputs, "actions");
            var batch = states.shape[0];
            var context = states.shape[1];
            var stateSize = states.shape[2];
            var actionSize = actions.shape[2];
            var tolerance = backendType == BackendType.CPU
                ? fixture.cpu_max_abs_tolerance
                : fixture.gpu_max_abs_tolerance;
            var layout = new ActionLayout(
                fixture.continuous_size, fixture.branches, fixture.action_low, fixture.action_high);
            Assert.That(layout.ActionSize, Is.EqualTo(actionSize), "layout disagrees with the graph's action width");
            Assert.That(layout.EnvironmentActionSize, Is.EqualTo(fixture.env_action_width));

            var window = new WindowContext(batch, context, stateSize, actionSize);
            window.Reset(fixture.initial_state.data);

            using var backend = new UnityInferenceBackend(FixtureModels.LoadModelAsset(model), backendType);
            foreach (var step in fixture.steps)
            {
                AssertInputs(window.Inputs(), step.inputs, tolerance, $"{model} step {step.index} input");
                var input = window.Inputs();
                using var statesTensor = new Tensor<float>(new TensorShape(batch, context, stateSize), input.States);
                using var actionsTensor = new Tensor<float>(new TensorShape(batch, context, actionSize), input.Actions);
                using var bosTensor = new Tensor<float>(new TensorShape(batch, context, 1), input.IsBos);
                using var maskTensor = new Tensor<float>(new TensorShape(batch, context), input.Mask);

                var raw = backend.Execute(statesTensor, actionsTensor, bosTensor, maskTensor);
                AssertClose(raw, step.raw_output.data, tolerance, $"{model} step {step.index} raw output");

                var decoded = ActionCodec.Decode(raw, batch, layout);
                AssertGameFacingViewsAgree(decoded, layout, batch, $"{model} step {step.index}");
                AssertClose(decoded.FeedbackAction, step.feedback_action.data, tolerance, $"{model} step {step.index} feedback");
                AssertClose(decoded.EnvironmentAction, step.environment_action.data, tolerance, $"{model} step {step.index} environment action");

                window.AfterActDiscardBos();
                window.Update(step.next_state.data, decoded.FeedbackAction, 0.0f);
                AssertInputs(window.Inputs(), step.updated_inputs, tolerance, $"{model} step {step.index} updated window");
            }
        }

        /// <summary>
        /// The accessors a game uses must say the same thing as the flat array the fixtures
        /// record. They are a different view of one decode, not a second decode.
        /// </summary>
        private static void AssertGameFacingViewsAgree(
            DecodedAction decoded, ActionLayout layout, int batch, string label)
        {
            var width = layout.EnvironmentActionSize;
            var branchBuffer = new int[layout.BranchSizes.Count];
            for (var row = 0; row < batch; row++)
            {
                var continuous = decoded.Continuous(row);
                Assert.That(continuous.Length, Is.EqualTo(layout.ContinuousSize), $"{label} continuous width");
                for (var index = 0; index < continuous.Length; index++)
                {
                    Assert.That(
                        continuous[index],
                        Is.EqualTo(decoded.EnvironmentAction[row * width + index]),
                        $"{label} continuous[{row}][{index}]");
                }

                decoded.CopyDiscrete(row, branchBuffer);
                for (var branch = 0; branch < layout.BranchSizes.Count; branch++)
                {
                    var flat = decoded.EnvironmentAction[row * width + layout.ContinuousSize + branch];
                    Assert.That(flat, Is.EqualTo(Math.Floor(flat)), $"{label} branch {branch} is not a whole number");
                    Assert.That(decoded.Discrete(row, branch), Is.EqualTo((int)flat), $"{label} discrete[{row}][{branch}]");
                    Assert.That(branchBuffer[branch], Is.EqualTo((int)flat), $"{label} copied discrete[{row}][{branch}]");
                    Assert.That(
                        decoded.Discrete(row, branch),
                        Is.InRange(0, layout.BranchSizes[branch] - 1),
                        $"{label} discrete[{row}][{branch}] is outside its branch");
                }
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
                Assert.That(float.IsInfinity(actual[index]), Is.False, $"{label}[{index}] is infinite.");
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
