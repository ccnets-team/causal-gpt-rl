using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// The engine layer's one rule: a worker holds a single output buffer, so a second
    /// schedule overwrites the first result. Every path that re-schedules has to say so
    /// rather than hand back a value belonging to another pass.
    /// </summary>
    public sealed class InferenceBackendTests
    {
        private static UnityInferenceBackend NewBackend(out Tensor<float>[] inputs)
        {
            var model = FixtureModels.All().First();
            var backend = new UnityInferenceBackend(FixtureModels.LoadModelAsset(model), BackendType.CPU);
            inputs = MakeInputs(backend);
            return backend;
        }

        private static Tensor<float>[] MakeInputs(UnityInferenceBackend backend)
        {
            return new[]
            {
                new Tensor<float>(
                    new TensorShape(backend.BatchSize, backend.ContextLength, backend.StateSize)),
                new Tensor<float>(
                    new TensorShape(backend.BatchSize, backend.ContextLength, backend.ActionSize)),
                new Tensor<float>(new TensorShape(backend.BatchSize, backend.ContextLength, 1)),
                new Tensor<float>(new TensorShape(backend.BatchSize, backend.ContextLength)),
            };
        }

        private static PendingExecution Schedule(UnityInferenceBackend backend, Tensor<float>[] inputs)
        {
            return backend.Schedule(inputs[0], inputs[1], inputs[2], inputs[3]);
        }

        [Test]
        public void RefusesASecondSchedule()
        {
            using var backend = NewBackend(out var first);
            Schedule(backend, first);
            var second = MakeInputs(backend);

            try
            {
                Assert.That(() => Schedule(backend, second), Throws.InvalidOperationException);
            }
            finally
            {
                foreach (var tensor in second) tensor.Dispose();
            }
        }

        [Test]
        public void RefusesABlockingExecuteWhileScheduled()
        {
            // Execute re-schedules the same worker, so it invalidates the pending result too.
            using var backend = NewBackend(out var first);
            Schedule(backend, first);
            var second = MakeInputs(backend);

            try
            {
                Assert.That(
                    () => backend.Execute(second[0], second[1], second[2], second[3]),
                    Throws.InvalidOperationException);
            }
            finally
            {
                foreach (var tensor in second) tensor.Dispose();
            }
        }

        [Test]
        public void ReadingTheResultFreesTheBackend()
        {
            using var backend = NewBackend(out var first);
            var pending = Schedule(backend, first);

            Assert.That(pending.GetResult(), Has.Length.EqualTo(backend.BatchSize * backend.ActionSize));
            Assert.That(() => pending.GetResult(), Throws.InvalidOperationException, "read twice");

            var second = MakeInputs(backend);
            Assert.DoesNotThrow(() => Schedule(backend, second).GetResult());
        }

        [Test]
        public void AbandoningAnExecutionFreesTheBackendAndKillsTheResult()
        {
            using var backend = NewBackend(out var first);
            var pending = Schedule(backend, first);
            pending.Dispose();

            // The backend may schedule again, which is precisely why the abandoned execution
            // must not read: that output buffer now belongs to the next pass.
            var second = MakeInputs(backend);
            Assert.DoesNotThrow(() => Schedule(backend, second).GetResult());
            Assert.That(() => pending.GetResult(), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => pending.IsDone, Throws.TypeOf<ObjectDisposedException>());
        }
    }
}
