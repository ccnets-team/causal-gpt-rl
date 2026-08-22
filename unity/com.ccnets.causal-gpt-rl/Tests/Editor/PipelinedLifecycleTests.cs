using System;
using System.Collections;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// A pipelined caller schedules the next action and then steps the environment with the
    /// action it collected a turn earlier, so the forward pass overlaps the physics instead of
    /// stalling in front of it. From the runner's side that is one thing: frames pass between
    /// <see cref="PolicyRunner.RequestAction"/> and <see cref="ActionRequest.GetAction"/>.
    ///
    /// This suite asks whether the shape needs anything the runner does not already have. It
    /// covers the four moments the gate names - the episode-start warm-up, a steady turn, a
    /// row ending while an action is in flight, and that row's first action afterwards - by
    /// replaying the recorded rollouts through the pipelined call order and holding the result
    /// to the same reference the serial order is held to.
    ///
    /// It deliberately does not assert on <see cref="ActionRequest.IsDone"/>. Whether the pass
    /// has landed by a given frame is a property of the machine, not of the call order, and
    /// asserting it makes the suite report load as a bug. GetAction waits if it has to.
    /// </summary>
    public sealed class PipelinedLifecycleTests
    {
        /// <summary>
        /// Frames between scheduling and collecting. Stands in for the environment step the
        /// pipeline exists to overlap - two, because one would often land on the frame the
        /// readback completes in, which is the case that never broke.
        /// </summary>
        private const int StepFrames = 2;

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
            public TensorFixture environment_action;
            public TensorFixture next_state;
        }

        [Serializable]
        private sealed class RolloutFixture
        {
            public TensorFixture initial_state;
            public float cpu_max_abs_tolerance;
            public float gpu_max_abs_tolerance;
            public RolloutStep[] steps;
        }

        /// <summary>
        /// The warm-up turn and every steady turn after it. Both runners are fed the same
        /// recorded observations; the only difference is that one collects its action at once
        /// and the other lets the environment step happen first.
        ///
        /// Exact equality, not a tolerance: same graph, same backend, same inputs. A tolerance
        /// here would hide the thing the test is for, which is the call order changing what
        /// the policy is asked.
        /// </summary>
        [UnityTest]
        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.CoroutineBackends))]
        public IEnumerator PipelinedTurnsProduceTheSerialActionSequence(string model, BackendType backendType)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");

            using var serial = Open(model, backendType);
            using var pipelined = Open(model, backendType);
            serial.Reset(fixture.initial_state.data);
            pipelined.Reset(fixture.initial_state.data);

            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";

                var serialAction = serial.Act();
                serial.Observe(step.next_state.data);

                var request = pipelined.RequestAction();
                for (var frame = 0; frame < StepFrames; frame++)
                {
                    yield return null;
                }
                var pipelinedAction = request.GetAction();
                pipelined.Observe(step.next_state.data);

                Assert.That(
                    pipelinedAction.EnvironmentAction, Is.EqualTo(serialAction.EnvironmentAction),
                    $"{label}: holding the action across the environment step changed it.");
                Assert.That(
                    pipelinedAction.FeedbackAction, Is.EqualTo(serialAction.FeedbackAction),
                    $"{label}: the action fed back into the window differs between the two orders.");
            }
        }

        /// <summary>
        /// A row ends while an action is in flight - which in a pipeline is where every
        /// termination happens, because the physics runs in exactly that window.
        ///
        /// The route that works is to collect the pending action and discard it for the ended
        /// rows, not to cancel it. The forward pass has already run either way, so collecting
        /// costs one readback and nothing else, and it leaves the runner in the state
        /// <see cref="PolicyRunner.ResetRows"/> wants. Held to the recorded reference so this
        /// is parity with the serial path, not merely a call sequence that does not throw.
        /// </summary>
        [UnityTest]
        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.RowResetCoroutineBackends))]
        public IEnumerator PipelinedRowResetMatchesTheReference(string model, BackendType backendType)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rowreset-expected.json");
            var tolerance = backendType == BackendType.CPU
                ? fixture.cpu_max_abs_tolerance
                : fixture.gpu_max_abs_tolerance;
            Assert.That(
                fixture.steps.Any(step => step.reset_rows != null && step.reset_rows.Length > 0),
                "This fixture resets no rows, so it does not cover what this test is for.");

            using var runner = Open(model, backendType);
            runner.Reset(fixture.initial_state.data);

            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index} pipelined";
                var request = runner.RequestAction();

                // The environment step, and with it the terminations, happen here.
                for (var frame = 0; frame < StepFrames; frame++)
                {
                    yield return null;
                }

                var action = request.GetAction();
                AssertClose(
                    action.EnvironmentAction, step.environment_action.data, tolerance,
                    $"{label} environment action");

                var resetRows = step.reset_rows ?? Array.Empty<int>();
                if (resetRows.Length > 0)
                {
                    runner.ResetRows(resetRows);
                }
                runner.Observe(step.next_state.data);
            }
        }

        /// <summary>
        /// The route that does not work, pinned so the boundary is written down somewhere it
        /// can fail.
        ///
        /// Cancelling returns the runner to Ready - every row holding a current observation -
        /// and <see cref="PolicyRunner.ResetRows"/> only runs between an action and the
        /// observations that follow it. So a caller that cancels a pending action has no way
        /// to retire the row whose episode just ended. Collecting the action instead costs a
        /// readback of a result nobody uses and leaves the runner where the reset belongs.
        ///
        /// Whole-batch termination is not affected: Reset is accepted from Ready.
        /// </summary>
        [Test]
        public void CancellingLeavesNoWayToRetireAnEndedRow()
        {
            var model = FixtureModels.WithRowReset().FirstOrDefault();
            Assert.That(model, Is.Not.Null, "No batched fixture is staged, so this covers nothing.");

            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rowreset-expected.json");
            using var runner = Open(model, BackendType.CPU);
            runner.Reset(fixture.initial_state.data);

            runner.RequestAction().Cancel();

            var refused = Assert.Throws<InvalidOperationException>(() => runner.ResetRows(new[] { 0 }));
            Assert.That(refused.Message, Does.Contain("ResetRows"));

            // The supported route, from the same starting point.
            runner.RequestAction().GetAction();
            Assert.DoesNotThrow(() => runner.ResetRows(new[] { 0 }));
        }

        private static PolicyRunner Open(string model, BackendType backendType)
        {
            return PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                backendType);
        }

        private static void AssertClose(float[] actual, float[] expected, float tolerance, string label)
        {
            Assert.That(actual, Has.Length.EqualTo(expected.Length), label);
            var maxAbs = 0.0f;
            var maxIndex = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                var error = Math.Abs(actual[index] - expected[index]);
                if (error > maxAbs)
                {
                    maxAbs = error;
                    maxIndex = index;
                }
            }
            Assert.That(maxAbs, Is.LessThanOrEqualTo(tolerance), $"{label} max_abs={maxAbs:R} at {maxIndex}");
        }
    }
}
