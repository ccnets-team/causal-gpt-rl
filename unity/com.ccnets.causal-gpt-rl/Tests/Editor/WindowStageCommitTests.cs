using System;
using System.Linq;
using NUnit.Framework;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// <see cref="WindowContext.Update"/> takes the observation that follows an action, so it
    /// cannot run until the environment has been stepped. The pass that produces the next
    /// action does not need that observation — the model never reads the staging slot — so
    /// coupling them is what forces the pass to wait for the world.
    ///
    /// <see cref="WindowContext.StageObservation"/> and
    /// <see cref="WindowContext.CommitFeedback"/> are the same bookkeeping, reachable a
    /// decision earlier. They are a second implementation of it, which is the risk: this
    /// suite pins them to the first one, byte for byte, over the recorded rollouts.
    ///
    /// The two are driven with different arguments on purpose. `Update` is told the state that
    /// comes next; the halves are told the state that is already here. Both end a turn with
    /// the same pair in the last slot the model reads, and that is the claim.
    /// </summary>
    public sealed class WindowStageCommitTests
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
            public TensorFixture feedback_action;
            public TensorFixture next_state;
        }

        [Serializable]
        private sealed class RolloutFixture
        {
            public TensorFixture initial_state;
            public RolloutStep[] steps;
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Models))]
        public void StageThenCommitMatchesUpdateAcrossARollout(string model)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            Assert.That(fixture.steps, Has.Length.GreaterThanOrEqualTo(10));

            var shape = Find(fixture.steps[0].inputs, "states").shape;
            var batch = shape[0];
            var actionSize = Find(fixture.steps[0].inputs, "actions").shape[2];

            var coupled = new WindowContext(batch, shape[1], shape[2], actionSize);
            var split = new WindowContext(batch, shape[1], shape[2], actionSize);
            coupled.Reset(fixture.initial_state.data);
            split.Reset(fixture.initial_state.data);

            var here = fixture.initial_state.data;
            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";
                var noRestarts = new float[batch];

                coupled.AfterActDiscardBos();
                coupled.Update(step.next_state.data, step.feedback_action.data, noRestarts);

                split.AfterActDiscardBos();
                split.StageObservation(here);
                split.CommitFeedback(step.feedback_action.data, noRestarts);

                AssertSame(coupled.Inputs(), split.Inputs(), label);
                here = step.next_state.data;
            }
        }

        /// <summary>
        /// The restart case, which is where the two halves could most plausibly disagree:
        /// `Update` copies the new observation into the visible slot itself when a row is
        /// starting an episode, and the halves have no such step. They are supposed not to
        /// need one, because staging happens before the roll rather than after it.
        /// </summary>
        [Test]
        public void StageThenCommitMatchesUpdateWhenRowsRestart()
        {
            var staged = FixtureModels.WithRowReset();
            Assert.That(staged, Is.Not.Empty, "No batched fixture is staged, so this covers nothing.");
            foreach (var model in staged)
            {
                StageThenCommitMatchesUpdateWhenRowsRestart(model);
            }
        }

        private static void StageThenCommitMatchesUpdateWhenRowsRestart(string model)
        {
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rowreset-expected.json");
            Assert.That(
                fixture.steps.Any(step => step.reset_rows != null && step.reset_rows.Length > 0),
                "This fixture resets no rows, so it does not cover what this test is for.");

            var shape = Find(fixture.steps[0].inputs, "states").shape;
            var batch = shape[0];
            var actionSize = Find(fixture.steps[0].inputs, "actions").shape[2];

            var coupled = new WindowContext(batch, shape[1], shape[2], actionSize);
            var split = new WindowContext(batch, shape[1], shape[2], actionSize);
            coupled.Reset(fixture.initial_state.data);
            split.Reset(fixture.initial_state.data);

            var here = fixture.initial_state.data;
            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";
                var resetRows = step.reset_rows ?? Array.Empty<int>();
                var restarting = new float[batch];
                foreach (var row in resetRows)
                {
                    restarting[row] = 1.0f;
                }

                var coupledFeedback = (float[])step.feedback_action.data.Clone();
                var splitFeedback = (float[])step.feedback_action.data.Clone();
                foreach (var row in resetRows)
                {
                    Array.Clear(coupledFeedback, row * actionSize, actionSize);
                    Array.Clear(splitFeedback, row * actionSize, actionSize);
                }

                coupled.AfterActDiscardBos();
                if (resetRows.Length > 0)
                {
                    coupled.ResetRows(resetRows);
                }
                coupled.Update(step.next_state.data, coupledFeedback, restarting);

                // A restarted row's episode begins at the observation the environment reports
                // after the reset, which for the coupled path is the one it passes to Update.
                var splitHere = (float[])here.Clone();
                foreach (var row in resetRows)
                {
                    Array.Copy(
                        step.next_state.data, row * shape[2], splitHere, row * shape[2], shape[2]);
                }

                split.AfterActDiscardBos();
                if (resetRows.Length > 0)
                {
                    split.ResetRows(resetRows);
                }
                split.StageObservation(splitHere);
                split.CommitFeedback(splitFeedback, restarting);

                AssertSame(coupled.Inputs(), split.Inputs(), label);
                here = step.next_state.data;
            }
        }

        private static void AssertSame(WindowInputs coupled, WindowInputs split, string label)
        {
            Assert.That(split.States, Is.EqualTo(coupled.States), $"{label} states");
            Assert.That(split.Actions, Is.EqualTo(coupled.Actions), $"{label} actions");
            Assert.That(split.IsBos, Is.EqualTo(coupled.IsBos), $"{label} is_bos");
            Assert.That(split.Mask, Is.EqualTo(coupled.Mask), $"{label} mask");
        }

        private static TensorFixture Find(TensorFixture[] values, string name)
        {
            var value = values.SingleOrDefault(item => item.name == name);
            Assert.That(value, Is.Not.Null, $"Missing tensor fixture '{name}'.");
            return value;
        }
    }
}
