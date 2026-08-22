using System;
using System.Collections;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// The assembled runner against the recorded reference. The component tests prove each
    /// piece; this proves the sequence a game actually writes — Reset, RequestAction,
    /// GetAction, ResetRows, Observe — reproduces the Python rollout step for step.
    /// </summary>
    public sealed class PolicyRunnerParityTests
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
            public TensorFixture feedback_action;
            public TensorFixture environment_action;
            public TensorFixture next_state;
        }

        [Serializable]
        private sealed class RolloutFixture
        {
            public TensorFixture initial_state;
            public int continuous_size;
            public int[] branches;
            public int env_action_width;
            public float cpu_max_abs_tolerance;
            public float gpu_max_abs_tolerance;
            public RolloutStep[] steps;
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Backends))]
        public void RolloutMatchesOnnxRuntime(string model, BackendType backendType)
        {
            Replay(model, backendType, "rollout-expected.json");
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.RowResetBackends))]
        public void RowResetRolloutMatchesOnnxRuntime(string model, BackendType backendType)
        {
            Replay(model, backendType, "rowreset-expected.json");
        }

        private static void Replay(string model, BackendType backendType, string fixtureName)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, fixtureName);
            var tolerance = backendType == BackendType.CPU
                ? fixture.cpu_max_abs_tolerance
                : fixture.gpu_max_abs_tolerance;

            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                backendType);

            Assert.That(runner.ActionSize, Is.EqualTo(runner.ActionLayout.ActionSize));
            Assert.That(runner.EnvironmentActionSize, Is.EqualTo(fixture.env_action_width));
            Assert.That(runner.BosCacheMode, Is.EqualTo("discard"));

            runner.Reset(fixture.initial_state.data);

            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";

                var request = runner.RequestAction();
                // The split exists so a caller can do its own work here; polling must not be
                // required for correctness, so read it straight away and check both paths agree.
                var action = request.GetAction();
                Assert.That(request.IsDone, Is.True, $"{label} request should be done once read");
                Assert.That(request.GetAction(), Is.SameAs(action), $"{label} second read differs");

                AssertClose(
                    action.EnvironmentAction, step.environment_action.data, tolerance,
                    $"{label} environment action");

                var resetRows = step.reset_rows ?? Array.Empty<int>();
                AssertFeedbackOutsideResetRows(action, step, resetRows, runner.ActionSize, tolerance, label);

                if (resetRows.Length > 0)
                {
                    runner.ResetRows(resetRows);
                }
                runner.Observe(step.next_state.data);
            }
        }

        /// <summary>
        /// The recorded feedback is what the window was fed, so an ended row reads as zeros
        /// there. The runner zeroes its own copy inside ResetRows instead of editing the
        /// action the caller is holding, so those rows are compared through the next steps'
        /// outputs rather than here — a wrongly carried action diverges the very next step.
        /// </summary>
        private static void AssertFeedbackOutsideResetRows(
            DecodedAction action,
            RolloutStep step,
            int[] resetRows,
            int actionSize,
            float tolerance,
            string label)
        {
            var rows = action.EnvironmentAction.Length / action.Layout.EnvironmentActionSize;
            for (var row = 0; row < rows; row++)
            {
                if (resetRows.Contains(row))
                {
                    continue;
                }
                for (var index = 0; index < actionSize; index++)
                {
                    var offset = row * actionSize + index;
                    Assert.That(
                        Math.Abs(action.FeedbackAction[offset] - step.feedback_action.data[offset]),
                        Is.LessThanOrEqualTo(tolerance),
                        $"{label} feedback[{row}][{index}]");
                }
            }
        }

        /// <summary>
        /// The reason the request is split in two. Nothing here asserts that the first poll
        /// says "not done" — a small graph on a warm backend may finish immediately — only
        /// that waiting across frames works and returns the same action a blocking read does.
        /// </summary>
        [UnityTest]
        public IEnumerator ActionSurvivesWaitingAcrossFrames()
        {
            var model = FixtureModels.All().First();
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.GPUCompute);

            runner.Reset(fixture.initial_state.data);
            var request = runner.RequestAction();

            // Yield unconditionally: a warm backend can finish before the first poll, and then
            // a `while (!IsDone)` loop would never cross a frame — leaving the thing this test
            // exists to prove (the output and its inputs survive the boundary) unexercised.
            var frames = 1;
            yield return null;
            while (!request.IsDone && frames < 600)
            {
                frames++;
                yield return null;
            }
            Assert.That(request.IsDone, Is.True, $"readback never completed within {frames} frames");

            var action = request.GetAction();
            AssertClose(
                action.EnvironmentAction,
                fixture.steps[0].environment_action.data,
                fixture.gpu_max_abs_tolerance,
                $"{model} action after waiting {frames} frame(s)");
        }

        [Test]
        public void RefusesResetRowsWhileAnActionIsInFlight()
        {
            // The counterexample this guard exists for: the reset would clear the rows and the
            // arriving action would then write its feedback straight back over them.
            using var runner = StartedRunner();
            runner.RequestAction();

            Assert.That(() => runner.ResetRows(new[] { 0 }), Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesResetWhileAnActionIsInFlight()
        {
            // Otherwise the next request schedules on the same worker and invalidates the
            // output the first request still points at.
            using var runner = StartedRunner();
            runner.RequestAction();

            Assert.That(
                () => runner.Reset(new float[runner.BatchSize * runner.StateSize]),
                Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesObserveWhileAnActionIsInFlight()
        {
            // An observation recorded before the action is read would be discarded when the
            // action lands, and the caller would never learn it was dropped.
            using var runner = StartedRunner();
            runner.RequestAction();

            Assert.That(
                () => runner.Observe(new float[runner.BatchSize * runner.StateSize]),
                Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesASecondObserveForTheSameStep()
        {
            // The first one closes the step; a second would be accepted and never reach the
            // window, so the caller would act on the observation before it.
            using var runner = StartedRunner();
            runner.RequestAction().GetAction();
            runner.Observe(new float[runner.BatchSize * runner.StateSize]);

            Assert.That(
                () => runner.Observe(new float[runner.BatchSize * runner.StateSize]),
                Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesReportingTheSameRowTwice()
        {
            // Exactly once per step. Overwriting would drop one of the two values with no
            // sign, and two components observing the same agent is a real way to get here.
            var model = FixtureModels.WithRowReset().First();
            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.CPU);

            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            runner.RequestAction().GetAction();
            runner.ObserveRow(0, new float[runner.StateSize]);

            Assert.That(
                () => runner.ObserveRow(0, new float[runner.StateSize]),
                Throws.InvalidOperationException);
            Assert.That(
                () => runner.Observe(new float[runner.BatchSize * runner.StateSize]),
                Throws.InvalidOperationException,
                "a whole-batch report after a partial one would overwrite what row 0 said");
        }

        [Test]
        public void RefusesResettingARowThatAlreadyReported()
        {
            // Otherwise the reset clears the report and the same row can be observed twice for
            // one action — the way around exactly-once.
            var model = FixtureModels.WithRowReset().First();
            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.CPU);

            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            runner.RequestAction().GetAction();
            runner.ObserveRow(0, new float[runner.StateSize]);

            Assert.That(() => runner.ResetRows(new[] { 0 }), Throws.InvalidOperationException);
            Assert.DoesNotThrow(
                () => runner.ResetRows(new[] { 1 }),
                "a row that has not reported yet is independent and may still end its episode");
        }

        [Test]
        public void LayoutCannotBeEditedThroughTheRunner()
        {
            // The decode clips against these bounds every step, and the validator already
            // refused anything but [-1, 1]. An editable layout would undo that check.
            using var runner = NewRunner();
            var layout = runner.ActionLayout;

            Assert.That(layout.Low, Is.Not.InstanceOf<float[]>());
            Assert.That(layout.High, Is.Not.InstanceOf<float[]>());
            Assert.That(layout.BranchSizes, Is.Not.InstanceOf<System.Collections.Generic.List<int>>());
        }

        [Test]
        public void RefusesEverythingAfterDispose()
        {
            var runner = StartedRunner();
            runner.Dispose();

            Assert.That(() => runner.RequestAction(), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => runner.Reset(null), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(
                () => runner.Observe(null),
                Throws.TypeOf<ObjectDisposedException>());
            Assert.That(
                () => runner.ObserveRow(-1, null),
                Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => runner.ResetRows(null), Throws.TypeOf<ObjectDisposedException>());
            Assert.DoesNotThrow(() => runner.Dispose(), "disposing twice should be harmless");
        }

        [Test]
        public void UnreadActionRequestDiesWithItsRunner()
        {
            var runner = StartedRunner();
            var request = runner.RequestAction();

            runner.Dispose();

            Assert.That(() => request.IsDone, Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => request.GetAction(), Throws.TypeOf<ObjectDisposedException>());
        }

        [Test]
        public void CompletedActionRequestOutlivesItsRunner()
        {
            var runner = StartedRunner();
            var request = runner.RequestAction();
            var action = request.GetAction();

            runner.Dispose();

            Assert.That(request.IsDone, Is.True);
            Assert.That(request.GetAction(), Is.SameAs(action));
        }

        [Test]
        public void ActMatchesTheSplitRequest()
        {
            using var blocking = StartedRunner();
            using var split = StartedRunner();

            var blockingAction = blocking.Act();
            var splitAction = split.RequestAction().GetAction();

            Assert.That(blockingAction.EnvironmentAction, Is.EqualTo(splitAction.EnvironmentAction));
            Assert.That(blockingAction.FeedbackAction, Is.EqualTo(splitAction.FeedbackAction));
        }

        [Test]
        public void ResetWhileReadyRestartsTheRunner()
        {
            var model = FixtureModels.All().First();
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            using var runner = NewRunner();
            using var fresh = NewRunner();

            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            runner.Reset(fixture.initial_state.data);
            fresh.Reset(fixture.initial_state.data);

            Assert.That(
                runner.Act().EnvironmentAction,
                Is.EqualTo(fresh.Act().EnvironmentAction));
        }

        [Test]
        public void ResetWhileAwaitingObservationsRestartsTheRunner()
        {
            var model = FixtureModels.All().First();
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            using var runner = StartedRunner();
            using var fresh = NewRunner();

            runner.Act();
            runner.Reset(fixture.initial_state.data);
            fresh.Reset(fixture.initial_state.data);

            Assert.That(
                runner.Act().EnvironmentAction,
                Is.EqualTo(fresh.Act().EnvironmentAction));
        }

        [Test]
        public void ConstructorRejectsConfigGraphMismatch()
        {
            var model = FixtureModels.All().First();
            var root = JObject.Parse(FixtureModels.LoadText(model, "config.json"));
            var wrongContext = root.Value<int>("context_length") - 1;
            root["context_length"] = wrongContext;
            ((JObject)root["model_config"])["context_length"] = wrongContext;
            var config = BundleConfig.FromJson(root.ToString());

            Assert.That(
                () => new PolicyRunner(config, FixtureModels.LoadModelAsset(model), BackendType.CPU),
                Throws.TypeOf<BundleValidationException>(),
                "the constructor must reject a valid config paired with the wrong graph");
        }

        private static PolicyRunner StartedRunner()
        {
            var runner = NewRunner();
            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            return runner;
        }

        [Test]
        public void RefusesActionBeforeReset()
        {
            using var runner = NewRunner();

            Assert.That(() => runner.RequestAction(), Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesActionWhileAStepIsOpen()
        {
            using var runner = NewRunner();
            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            runner.RequestAction().GetAction();

            // The action was taken but no observation followed it, so acting again would run
            // the policy on a state it has already answered.
            Assert.That(() => runner.RequestAction(), Throws.InvalidOperationException);
        }

        [Test]
        public void RefusesActionWhileARowIsUnobserved()
        {
            var model = FixtureModels.WithRowReset().First();
            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.CPU);

            runner.Reset(new float[runner.BatchSize * runner.StateSize]);
            runner.RequestAction().GetAction();
            for (var row = 1; row < runner.BatchSize; row++)
            {
                runner.ObserveRow(row, new float[runner.StateSize]);
            }

            Assert.That(() => runner.RequestAction(), Throws.InvalidOperationException,
                "row 0 never reported, so the step is still open");
        }

        [Test]
        public void RowWiseObservationMatchesWholeBatch()
        {
            var model = FixtureModels.WithRowReset().First();
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");

            var whole = ActionsFrom(model, fixture, rowWise: false);
            var rowWise = ActionsFrom(model, fixture, rowWise: true);

            Assert.That(rowWise, Is.EqualTo(whole), "reporting row by row diverged from one batched call");
        }

        private static float[] ActionsFrom(string model, RolloutFixture fixture, bool rowWise)
        {
            using var runner = PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.CPU);

            runner.Reset(fixture.initial_state.data);
            var last = Array.Empty<float>();
            foreach (var step in fixture.steps)
            {
                last = runner.RequestAction().GetAction().EnvironmentAction;
                if (!rowWise)
                {
                    runner.Observe(step.next_state.data);
                    continue;
                }

                var one = new float[runner.StateSize];
                for (var row = 0; row < runner.BatchSize; row++)
                {
                    Array.Copy(step.next_state.data, row * runner.StateSize, one, 0, runner.StateSize);
                    runner.ObserveRow(row, one);
                }
            }
            return last;
        }

        private static PolicyRunner NewRunner()
        {
            var model = FixtureModels.All().First();
            return PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                BackendType.CPU);
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
