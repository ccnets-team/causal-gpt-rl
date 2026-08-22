using System;
using System.Collections;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// The pipelined turn. This policy generates its action from the trajectory so far and
    /// never reads the newest observation - the window keeps one slot more than the model - so
    /// the pass for the next decision does not have to wait for the environment to produce
    /// one. <see cref="PolicyRunner.StageObservation"/> is what lets a caller say so: the
    /// observation the in-flight action belongs to is supplied before the read, the read
    /// closes the pair, and the next pass is scheduled immediately and runs under the step.
    ///
    /// What this suite is for is that the two orders must ask the policy the same question.
    /// A turn shape that pairs the wrong action with an observation still runs, still returns
    /// actions of the right width, and quietly conditions the model on a trajectory that never
    /// happened - measured on the Humanoid bundle as a fall at the same decision every episode.
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
            public TensorFixture environment_action;
            public TensorFixture next_state;
        }

        [Serializable]
        private sealed class RolloutFixture
        {
            public TensorFixture initial_state;
            public RolloutStep[] steps;
        }

        /// <summary>
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

            // One decision of warm-up: the first action of an episode cannot have been
            // scheduled under a previous step, because there was not one.
            var request = pipelined.RequestAction();
            var here = fixture.initial_state.data;

            foreach (var step in fixture.steps)
            {
                var label = $"{model} step {step.index}";

                var serialAction = serial.Act();
                serial.Observe(step.next_state.data);

                // The environment step the pass has been running under.
                for (var frame = 0; frame < StepFrames; frame++)
                {
                    yield return null;
                }

                pipelined.StageObservation(here);
                var pipelinedAction = request.GetAction();
                request = pipelined.RequestAction();
                here = step.next_state.data;

                Assert.That(
                    pipelinedAction.EnvironmentAction, Is.EqualTo(serialAction.EnvironmentAction),
                    $"{label}: the pipelined turn asked the policy a different question.");
                Assert.That(
                    pipelinedAction.FeedbackAction, Is.EqualTo(serialAction.FeedbackAction),
                    $"{label}: the action fed back into the window differs between the orders.");
            }

            request.Cancel();
        }

        /// <summary>
        /// Staging is what separates the two orders, and mixing them has to be refused rather
        /// than silently producing a window nobody meant. <see cref="PolicyRunner.Observe"/>
        /// reports the observation that FOLLOWS an action; staging reports the one the action
        /// was taken AT. A runner that accepted both in one turn would advance twice.
        /// </summary>
        [Test]
        public void TheTwoOrdersCannotBeMixedWithinATurn()
        {
            var model = FixtureModels.All().First();
            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rollout-expected.json");
            using var runner = Open(model, BackendType.CPU);
            runner.Reset(fixture.initial_state.data);

            var request = runner.RequestAction();
            runner.StageObservation(fixture.initial_state.data);

            // Reading a staged action leaves nothing outstanding, so the serial report has
            // nothing to report against.
            request.GetAction();
            var refused = Assert.Throws<InvalidOperationException>(
                () => runner.Observe(fixture.steps[0].next_state.data));
            Assert.That(refused.Message, Does.Contain("Observe"));

            // And the serial order cannot stage: there is no action in flight to pair with.
            runner.Act();
            Assert.Throws<InvalidOperationException>(
                () => runner.StageObservation(fixture.steps[0].next_state.data));
        }

        /// <summary>
        /// What the pipelined turn cannot express yet, pinned so the boundary is written down
        /// somewhere it can fail.
        ///
        /// <see cref="PolicyRunner.ResetRows"/> runs between an action and the observations
        /// that follow it, and the pipelined turn never enters that state - it goes from the
        /// staged read straight back to ready. A whole-batch restart is unaffected, since
        /// cancelling the pending action leaves the runner where <see cref="PolicyRunner.Reset"/>
        /// is accepted; a batched scene retiring one row of several has to run that turn
        /// serially.
        /// </summary>
        [Test]
        public void RetiringOneRowIsNotExpressibleInThePipelinedTurn()
        {
            var model = FixtureModels.WithRowReset().FirstOrDefault();
            Assert.That(model, Is.Not.Null, "No batched fixture is staged, so this covers nothing.");

            var fixture = FixtureModels.LoadJson<RolloutFixture>(model, "rowreset-expected.json");
            using var runner = Open(model, BackendType.CPU);
            runner.Reset(fixture.initial_state.data);

            var request = runner.RequestAction();
            runner.StageObservation(fixture.initial_state.data);
            request.GetAction();

            var refused = Assert.Throws<InvalidOperationException>(() => runner.ResetRows(new[] { 0 }));
            Assert.That(refused.Message, Does.Contain("ResetRows"));

            // The whole-batch restart the single-agent case uses is fine.
            var pending = runner.RequestAction();
            pending.Cancel();
            Assert.DoesNotThrow(() => runner.Reset(fixture.initial_state.data));
        }

        private static PolicyRunner Open(string model, BackendType backendType)
        {
            return PolicyRunner.Load(
                FixtureModels.LoadModelAsset(model),
                FixtureModels.LoadText(model, "config.json"),
                backendType);
        }
    }
}
