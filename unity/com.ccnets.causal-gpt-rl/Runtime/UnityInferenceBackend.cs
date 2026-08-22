using System;
using Unity.InferenceEngine;

namespace CCNets.CausalGPTRL
{
    /// <summary>
    /// One scheduled forward pass whose result has not been read back yet. Keeping the
    /// schedule and the readback apart is the point: a blocking call would stall the main
    /// thread for the whole inference, and a game cannot spend a frame that way.
    /// </summary>
    internal sealed class PendingExecution : IDisposable
    {
        private readonly Tensor<float> _output;
        private Tensor<float>[] _inputs;
        private Stage _stage = Stage.Active;

        private enum Stage
        {
            /// <summary>Scheduled; the result has not been read.</summary>
            Active,

            /// <summary>The result was read; the inputs are released.</summary>
            Consumed,

            /// <summary>Abandoned unread; the backend is free to schedule again.</summary>
            Disposed,
        }

        internal PendingExecution(Tensor<float> output, Tensor<float>[] inputs)
        {
            _output = output;
            _inputs = inputs;
            _output.ReadbackRequest();
        }

        /// <summary>
        /// True once the forward pass has finished, so reading the result does not wait on it.
        ///
        /// It does not mean the scheduled readback is still collectable: on a GPU backend that
        /// readback is only valid inside the frame it completes in, and this stays true
        /// afterwards. <see cref="GetResult"/> covers that case rather than the caller.
        /// </summary>
        public bool IsDone
        {
            get
            {
                RequireNotDisposed();
                return _output.IsReadbackRequestDone();
            }
        }

        /// <summary>True once the backend may schedule again.</summary>
        internal bool IsSettled => _stage != Stage.Active;

        /// <summary>
        /// Reads the result. Blocks until the pass is <see cref="IsDone"/>, so a caller that
        /// cannot afford a stall should poll first. Collecting it in a later frame than the
        /// one it became ready in is fine and costs a readback, not another pass.
        /// </summary>
        public float[] GetResult()
        {
            RequireNotDisposed();
            if (_stage == Stage.Consumed)
            {
                throw new InvalidOperationException("This execution's result was already read.");
            }
            _stage = Stage.Consumed;

            try
            {
                return Download();
            }
            finally
            {
                ReleaseInputs();
            }
        }

        /// <summary>
        /// Reads the output, retrying once when the scheduled readback has expired.
        ///
        /// A GPU readback request is only valid inside the frame it completes in; collect it
        /// later and it throws "Cannot access the data as it is not available" — with IsDone
        /// still true, because the request did finish. Nothing about the result is lost: the
        /// backend refuses to schedule again while this execution is unread, so the output
        /// buffer still holds it, and the second read fetches it directly. That read is a copy
        /// of finished work, so the pass still overlapped whatever the caller did meanwhile.
        ///
        /// The retry is what makes the split request safe to use at all. Without it a caller
        /// can only collect inside one frame it is never told about, and every other schedule
        /// is a decision length of physics thrown away.
        /// </summary>
        private float[] Download()
        {
            try
            {
                return DownloadOnce();
            }
            catch (InvalidOperationException)
            {
                return DownloadOnce();
            }
        }

        private float[] DownloadOnce()
        {
            using var cpuOutput = _output.ReadbackAndClone() as Tensor<float>;
            if (cpuOutput == null)
            {
                throw new InvalidOperationException("Could not read model output 'action' as float32.");
            }
            return cpuOutput.DownloadToArray();
        }

        /// <summary>
        /// Abandons the execution: its inputs are released and its result can no longer be
        /// read. The backend is free to schedule again, which is exactly why reading afterwards
        /// must fail — the output buffer that reads belongs to the next execution by then.
        /// </summary>
        public void Dispose()
        {
            if (_stage == Stage.Active)
            {
                _stage = Stage.Disposed;
            }
            ReleaseInputs();
        }

        /// <summary>
        /// Throws unless the result is still there to collect. The runner checks it before its
        /// own state: a request that was given up on, or whose read already failed, has to
        /// read as that rather than as whatever the runner has moved on to since.
        ///
        /// A successfully read result never arrives here — the request caches it and never
        /// asks a second time — so Consumed can only mean the read threw.
        /// </summary>
        internal void RequireLive()
        {
            RequireNotDisposed();
            if (_stage == Stage.Consumed)
            {
                throw new InvalidOperationException(
                    "This action request's read failed, and the runner was returned to where it " +
                    "stood before it. Request another action.");
            }
        }

        private void RequireNotDisposed()
        {
            if (_stage == Stage.Disposed)
            {
                // ActionRequest, not this type: PendingExecution is internal, so naming it
                // would point a caller at something outside the public surface.
                throw new ObjectDisposedException(nameof(ActionRequest));
            }
        }

        private void ReleaseInputs()
        {
            if (_inputs == null)
            {
                return;
            }
            foreach (var input in _inputs)
            {
                input?.Dispose();
            }
            _inputs = null;
        }
    }

    /// <summary>
    /// Wrapper around Unity Inference Engine. Causal window management and action decoding
    /// deliberately live outside this class.
    ///
    /// Internal: the supported surface is PolicyRunner. The performance harness reaches this
    /// layer through InternalsVisibleTo rather than by making it public, so its ownership and
    /// disposal rules stay ours to change.
    /// </summary>
    internal sealed class UnityInferenceBackend : IDisposable
    {
        private readonly Worker _worker;
        private PendingExecution _inFlight;

        public UnityInferenceBackend(ModelAsset modelAsset, BackendType backendType)
        {
            if (modelAsset == null)
            {
                throw new ArgumentNullException(nameof(modelAsset));
            }

            var model = ModelLoader.Load(modelAsset);
            ReadShapes(model);
            _worker = new Worker(model, backendType);
        }

        /// <summary>Rows the graph was exported for. Fixed — the batch is baked into the file.</summary>
        public int BatchSize { get; private set; }

        public int ContextLength { get; private set; }
        public int StateSize { get; private set; }

        /// <summary>The model's action width: continuous columns plus every branch's logits.</summary>
        public int ActionSize { get; private set; }

        /// <summary>
        /// Schedules the pass without reading the result back, and TAKES OWNERSHIP of the
        /// four input tensors.
        ///
        /// The ownership is not a convenience. Disposing a CPU tensor completes the job it
        /// belongs to (`BurstTensorData.Dispose` calls `CompleteAllPendingOperations` on the
        /// main thread), so releasing the inputs right after scheduling would block on the
        /// inference this method exists to avoid blocking on. They are released once the
        /// result is read, or when the execution is disposed unread.
        ///
        /// Only one execution may be in flight per backend: the worker reuses its output
        /// buffer, so a second schedule invalidates the first result.
        /// </summary>
        public PendingExecution Schedule(
            Tensor<float> states,
            Tensor<float> actions,
            Tensor<float> isBos,
            Tensor<float> mask)
        {
            EnsureIdle();

            var output = ScheduleInternal(states, actions, isBos, mask);
            _inFlight = new PendingExecution(output, new[] { states, actions, isBos, mask });
            return _inFlight;
        }

        /// <summary>
        /// Schedules the pass and blocks until the result is on the CPU.
        ///
        /// Unlike Schedule, the input tensors stay the caller's: this returns only after the
        /// readback, so there is no window in which they must outlive the call, and the
        /// callers that use it reuse their tensors across iterations.
        /// </summary>
        public float[] Execute(
            Tensor<float> states,
            Tensor<float> actions,
            Tensor<float> isBos,
            Tensor<float> mask)
        {
            // Same guard as Schedule: this also re-schedules the worker, so an unread pending
            // execution would be reading whatever this call leaves in the output buffer.
            EnsureIdle();

            var output = ScheduleInternal(states, actions, isBos, mask);
            using var cpuOutput = output.ReadbackAndClone() as Tensor<float>;
            if (cpuOutput == null)
            {
                throw new InvalidOperationException("Could not read model output 'action' as float32.");
            }
            return cpuOutput.DownloadToArray();
        }

        private void EnsureIdle()
        {
            if (_inFlight != null && !_inFlight.IsSettled)
            {
                throw new InvalidOperationException(
                    "This backend already has an execution in flight; read or dispose it first.");
            }
        }

        private Tensor<float> ScheduleInternal(
            Tensor<float> states,
            Tensor<float> actions,
            Tensor<float> isBos,
            Tensor<float> mask)
        {
            if (states == null) throw new ArgumentNullException(nameof(states));
            if (actions == null) throw new ArgumentNullException(nameof(actions));
            if (isBos == null) throw new ArgumentNullException(nameof(isBos));
            if (mask == null) throw new ArgumentNullException(nameof(mask));

            _worker.SetInput("states", states);
            _worker.SetInput("actions", actions);
            _worker.SetInput("is_bos", isBos);
            _worker.SetInput("mask", mask);
            _worker.Schedule();

            var output = _worker.PeekOutput("action") as Tensor<float>;
            if (output == null)
            {
                throw new InvalidOperationException("Model output 'action' is missing or is not float32.");
            }
            return output;
        }

        /// <summary>
        /// The inputs this runtime supplies, and the only ones a graph may declare.
        /// </summary>
        private static readonly string[] ExpectedInputs = { "states", "actions", "is_bos", "mask" };

        /// <summary>
        /// Reads the four input shapes off the graph so a bundle can be checked against the
        /// file it claims to describe. Every exported shape is static (no dynamic axes), and
        /// a graph that is not is refused rather than guessed at.
        /// </summary>
        private void ReadShapes(Model model)
        {
            // An unexpected extra input would never be set, so the graph would run on whatever
            // the engine defaults it to rather than failing.
            foreach (var input in model.inputs)
            {
                if (Array.IndexOf(ExpectedInputs, input.name) < 0)
                {
                    throw new BundleValidationException(
                        $"Graph declares an input '{input.name}' this runtime does not supply; " +
                        $"expected exactly {string.Join(", ", ExpectedInputs)}.");
                }
            }

            var states = RequireInput(model, "states", 3);
            var actions = RequireInput(model, "actions", 3);
            var isBos = RequireInput(model, "is_bos", 3);
            var mask = RequireInput(model, "mask", 2);

            BatchSize = states[0];
            ContextLength = states[1];
            StateSize = states[2];
            ActionSize = actions[2];

            if (BatchSize < 1 || ContextLength < 1 || StateSize < 1 || ActionSize < 1)
            {
                throw new BundleValidationException(
                    $"Graph declares a non-positive extent: batch={BatchSize}, context={ContextLength}, " +
                    $"state={StateSize}, action={ActionSize}.");
            }

            // The four inputs are read as one window, so they have to agree on how many rows
            // and how many steps that window holds. A graph whose axes disagree would be fed
            // tensors this runtime builds from `states` alone.
            RequireAxis("actions", actions, 0, BatchSize, "batch");
            RequireAxis("actions", actions, 1, ContextLength, "context length");
            RequireAxis("is_bos", isBos, 0, BatchSize, "batch");
            RequireAxis("is_bos", isBos, 1, ContextLength, "context length");
            RequireAxis("is_bos", isBos, 2, 1, "channel");
            RequireAxis("mask", mask, 0, BatchSize, "batch");
            RequireAxis("mask", mask, 1, ContextLength, "context length");

            RequireOutput(model, "action");
        }

        private static void RequireAxis(string name, int[] shape, int axis, int expected, string label)
        {
            if (shape[axis] == expected)
            {
                return;
            }
            throw new BundleValidationException(
                $"Graph input '{name}' declares {label} {shape[axis]}; the window is built for {expected}.");
        }

        /// <summary>
        /// Confirms the graph names the output this runtime reads. Only its presence can be
        /// checked here — `Model.Output` carries a name and an index, not a shape or a dtype,
        /// so the width is verified at the first decode instead (`ActionCodec.Decode` compares
        /// the returned length against the declared action size).
        /// </summary>
        private static void RequireOutput(Model model, string name)
        {
            foreach (var output in model.outputs)
            {
                if (output.name == name)
                {
                    return;
                }
            }

            throw new BundleValidationException(
                $"Graph is missing output '{name}'; this runtime reads the action from it.");
        }

        private static int[] RequireInput(Model model, string name, int rank)
        {
            foreach (var input in model.inputs)
            {
                if (input.name != name)
                {
                    continue;
                }
                if (input.dataType != DataType.Float)
                {
                    throw new BundleValidationException(
                        $"Graph input '{name}' is {input.dataType}; this runtime feeds float32.");
                }
                return RequireStaticShape(name, input.shape, rank);
            }

            throw new BundleValidationException(
                $"Graph is missing input '{name}'; expected states, actions, is_bos and mask.");
        }

        private static int[] RequireStaticShape(string name, DynamicTensorShape declared, int rank)
        {
            if (!declared.IsStatic())
            {
                throw new BundleValidationException(
                    $"Graph tensor '{name}' has a dynamic shape; this runtime sizes its window " +
                    "from the graph and cannot do that against an unknown extent.");
            }

            var shape = declared.ToTensorShape();
            if (shape.rank != rank)
            {
                throw new BundleValidationException(
                    $"Graph tensor '{name}' has rank {shape.rank}; expected {rank}.");
            }

            var dimensions = new int[rank];
            for (var axis = 0; axis < rank; axis++)
            {
                dimensions[axis] = shape[axis];
            }
            return dimensions;
        }

        public void Dispose()
        {
            _inFlight?.Dispose();
            _worker.Dispose();
        }
    }
}
