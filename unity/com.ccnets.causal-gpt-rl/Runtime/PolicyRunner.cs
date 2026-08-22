using System;
using System.Collections.Generic;
using Unity.InferenceEngine;

namespace CCNets.CausalGPTRL
{
    /// <summary>
    /// Represents a scheduled action awaiting readback. Poll <see cref="IsDone"/> and call
    /// <see cref="GetAction"/> when the result is needed, allowing the game to perform other
    /// work while inference runs.
    /// </summary>
    public sealed class ActionRequest
    {
        private readonly PolicyRunner _runner;
        private readonly PendingExecution _execution;
        private DecodedAction _action;

        internal ActionRequest(PolicyRunner runner, PendingExecution execution)
        {
            _runner = runner;
            _execution = execution;
        }

        /// <summary>
        /// True once the result can be read without blocking. A result that has already been
        /// read stays available even after its runner is disposed; an unread request is
        /// invalidated when its runner is disposed.
        /// </summary>
        public bool IsDone => _action != null || _execution.IsDone;

        /// <summary>
        /// Reads the result back, decodes it, and advances the runner past this action.
        /// Blocks if the result is not ready. Calling it twice returns the same action. Once
        /// returned, that cached action remains valid even if the runner is later disposed.
        /// </summary>
        public DecodedAction GetAction()
        {
            if (_action == null)
            {
                _action = _runner.CompleteAction(_execution);
            }
            return _action;
        }
    }

    /// <summary>
    /// Runs a Causal GPT-RL policy inside Unity by managing the rolling context, invoking the
    /// inference engine, and decoding its output. It follows the same call sequence as the
    /// Python `PolicyRunner`.
    ///
    /// A step is `Observe` → `RequestAction` → `GetAction` → apply → `ResetRows` for completed
    /// rows → `Observe` again. Observations are supplied as flat vectors and must use the same
    /// packing order as the trajectories used to create the bundle.
    /// </summary>
    public sealed class PolicyRunner : IDisposable
    {
        private readonly UnityInferenceBackend _backend;
        private readonly WindowContext _window;
        private readonly bool[] _observed;
        private readonly float[] _isBos;
        private readonly float[] _observations;
        private readonly float[] _feedback;
        private ActionRequest _inFlight;
        private State _state = State.NeedsReset;

        /// <summary>
        /// Where the runner is in a step. Every public call names the states it accepts, so a
        /// wrong order fails with a reason instead of quietly serving the policy a stale
        /// observation, or reviving the last action of an agent that already left.
        /// </summary>
        private enum State
        {
            /// <summary>Nothing observed yet.</summary>
            NeedsReset,

            /// <summary>Every row has a current observation; an action can be requested.</summary>
            Ready,

            /// <summary>An action is scheduled and not read back.</summary>
            InFlight,

            /// <summary>The action was taken; rows must report what followed it.</summary>
            AwaitingObservations,

            Disposed,
        }

        /// <summary>Opens a bundle: parses its config, validates it, and pairs it with the graph.</summary>
        public static PolicyRunner Load(ModelAsset policy, string configJson, BackendType backend)
        {
            return new PolicyRunner(BundleConfig.FromJson(configJson), policy, backend);
        }

        public PolicyRunner(BundleConfig config, ModelAsset policy, BackendType backend)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            BundleValidator.Validate(config);

            // The config and the graph are separate files and can be paired by mistake. Left
            // unchecked, the window would be built to one width and fed by another. Validate
            // before taking ownership: a constructor that throws is never disposed, so the
            // worker inside would leak.
            var opened = new UnityInferenceBackend(policy, backend);
            try
            {
                BundleValidator.ValidateGraph(
                    config, opened.BatchSize, opened.ContextLength, opened.StateSize, opened.ActionSize);
            }
            catch
            {
                opened.Dispose();
                throw;
            }
            _backend = opened;

            Config = config;
            ActionLayout = ActionLayout.FromConfig(config);
            _window = new WindowContext(BatchSize, ContextLength, StateSize, ActionSize);
            _observed = new bool[BatchSize];
            _isBos = new float[BatchSize];
            _observations = new float[BatchSize * StateSize];
            _feedback = new float[BatchSize * ActionSize];
        }

        public BundleConfig Config { get; }
        public ActionLayout ActionLayout { get; }

        /// <summary>Number of rows supported by the graph. The value is fixed in the exported model.</summary>
        public int BatchSize => _backend.BatchSize;

        public int ContextLength => _backend.ContextLength;
        public int StateSize => _backend.StateSize;

        /// <summary>The model's action width: continuous columns plus every branch's logits.</summary>
        public int ActionSize => _backend.ActionSize;

        /// <summary>The environment action width: continuous columns plus one index per branch.</summary>
        public int EnvironmentActionSize => ActionLayout.EnvironmentActionSize;

        public string BosCacheMode => Config.BosCacheMode;

        /// <summary>Starts every row on a fresh episode from the given observations.</summary>
        public void Reset(float[] observations)
        {
            RequireNotDisposed();
            RequireLength(observations, BatchSize * StateSize, nameof(observations));
            // Not while an action is in flight: that execution would be decoded against a
            // window it never saw, and its output buffer is about to be reused.
            Require(nameof(Reset), State.NeedsReset, State.Ready, State.AwaitingObservations);

            _window.Reset(observations);
            Array.Clear(_feedback, 0, _feedback.Length);
            Array.Clear(_isBos, 0, _isBos.Length);
            for (var row = 0; row < BatchSize; row++)
            {
                _observed[row] = true;
            }
            _inFlight = null;
            _state = State.Ready;
        }

        /// <summary>Records the observation that follows the previously emitted action.</summary>
        public void Observe(float[] observations)
        {
            RequireNotDisposed();
            RequireLength(observations, BatchSize * StateSize, nameof(observations));
            Require(nameof(Observe), State.AwaitingObservations);
            for (var row = 0; row < BatchSize; row++)
            {
                if (_observed[row])
                {
                    throw new InvalidOperationException(
                        $"Row {row} already reported this step; observing it twice would silently " +
                        "drop one of the two values. Report each row once per step.");
                }
            }

            Array.Copy(observations, _observations, _observations.Length);
            for (var row = 0; row < BatchSize; row++)
            {
                _observed[row] = true;
            }
            AdvanceIfComplete();
        }

        /// <summary>
        /// Records one row's observation. Agents report themselves, so a batched scene fills
        /// the vector row by row; the window advances once every row has reported.
        /// </summary>
        public void ObserveRow(int row, float[] observation)
        {
            RequireNotDisposed();
            RequireRow(row);
            RequireLength(observation, StateSize, nameof(observation));
            Require(nameof(ObserveRow), State.AwaitingObservations);
            if (_observed[row])
            {
                throw new InvalidOperationException(
                    $"Row {row} already reported this step; observing it twice would silently drop " +
                    "one of the two values. Report each row once per step.");
            }

            Array.Copy(observation, 0, _observations, row * StateSize, StateSize);
            _observed[row] = true;
            AdvanceIfComplete();
        }

        /// <summary>
        /// Ends the episode for the specified rows and clears their context and carried
        /// actions. Call this after applying the action that ended each episode.
        /// </summary>
        public void ResetRows(IReadOnlyList<int> rows)
        {
            RequireNotDisposed();
            if (rows == null) throw new ArgumentNullException(nameof(rows));
            // Only between the action and the observations that follow it. Called while an
            // action is in flight, the cleared rows would be overwritten by that action's
            // feedback, and a departed agent would go on steering whoever reuses the row.
            Require(nameof(ResetRows), State.AwaitingObservations);
            // A row that already reported has begun its next step; resetting it now would
            // clear the report and let the same row be observed twice for one action. Rows are
            // independent, so resetting a row that has not reported yet stays fine.
            foreach (var row in rows)
            {
                if (row >= 0 && row < BatchSize && _observed[row])
                {
                    throw new InvalidOperationException(
                        $"Row {row} already reported this step, so it cannot be reset here. " +
                        "Reset the rows whose episode ended before observing them.");
                }
            }

            _window.ResetRows(rows);
            foreach (var row in rows)
            {
                Array.Clear(_feedback, row * ActionSize, ActionSize);
                _isBos[row] = 1.0f;
                _observed[row] = false;
            }
        }

        /// <summary>
        /// Schedules the next action without waiting for it. One request may be in flight at
        /// a time: the engine reuses its output buffer, so a second schedule would overwrite
        /// the first result.
        /// </summary>
        public ActionRequest RequestAction()
        {
            Require(nameof(RequestAction), State.Ready);
            for (var row = 0; row < BatchSize; row++)
            {
                if (!_observed[row])
                {
                    throw new InvalidOperationException(
                        $"Row {row} was not observed since its last action; the policy would act on a " +
                        "stale observation.");
                }
            }

            // Not `using`: on success the backend takes ownership and releases them after the
            // readback. Disposing a CPU tensor completes the job it belongs to, so releasing
            // them here would block on the very inference this method exists not to wait for.
            // On failure nobody has taken them yet, hence the catch.
            var inputs = _window.Inputs();
            Tensor<float> states = null, actions = null, isBos = null, mask = null;
            try
            {
                states = new Tensor<float>(new TensorShape(BatchSize, ContextLength, StateSize), inputs.States);
                actions = new Tensor<float>(new TensorShape(BatchSize, ContextLength, ActionSize), inputs.Actions);
                isBos = new Tensor<float>(new TensorShape(BatchSize, ContextLength, 1), inputs.IsBos);
                mask = new Tensor<float>(new TensorShape(BatchSize, ContextLength), inputs.Mask);

                _inFlight = new ActionRequest(this, _backend.Schedule(states, actions, isBos, mask));
            }
            catch
            {
                states?.Dispose();
                actions?.Dispose();
                isBos?.Dispose();
                mask?.Dispose();
                throw;
            }

            _state = State.InFlight;
            return _inFlight;
        }

        /// <summary>Schedules an action and waits for it. The Python runner's `act`.</summary>
        public DecodedAction Act()
        {
            return RequestAction().GetAction();
        }

        internal DecodedAction CompleteAction(PendingExecution execution)
        {
            Require("GetAction", State.InFlight);

            var raw = execution.GetResult();
            var decoded = ActionCodec.Decode(raw, BatchSize, ActionLayout);

            // The episode-start token is used for this action and then dropped from every
            // later attention mask; the bundle's bos_cache_mode names that convention.
            _window.AfterActDiscardBos();

            // The window does not advance here. Rows that end because of this action are
            // wiped first, and only the next observation completes the step — which is also
            // why every row is marked unobserved now.
            // Copy rather than alias: ResetRows clears the ended rows' carried action, and
            // doing that through the caller's DecodedAction would edit a value it still holds.
            Array.Copy(decoded.FeedbackAction, _feedback, _feedback.Length);
            for (var row = 0; row < BatchSize; row++)
            {
                _observed[row] = false;
            }
            _state = State.AwaitingObservations;
            _inFlight = null;
            return decoded;
        }

        /// <summary>
        /// Advances the window as soon as the last row reports. Doing it here rather than at
        /// RequestAction keeps the roll — a memmove over the whole context — off the path
        /// between the final observation and the dispatch.
        /// </summary>
        private void AdvanceIfComplete()
        {
            for (var row = 0; row < BatchSize; row++)
            {
                if (!_observed[row])
                {
                    return;
                }
            }

            _window.Update(_observations, _feedback, _isBos);
            Array.Clear(_isBos, 0, _isBos.Length);
            _state = State.Ready;
        }

        public void Dispose()
        {
            if (_state == State.Disposed)
            {
                return;
            }
            _state = State.Disposed;
            _inFlight = null;
            _backend.Dispose();
        }

        private void Require(string call, params State[] allowed)
        {
            RequireNotDisposed();
            if (Array.IndexOf(allowed, _state) >= 0)
            {
                return;
            }

            string reason;
            switch (_state)
            {
                case State.NeedsReset:
                    reason = "call Reset(observations) first";
                    break;
                case State.Ready:
                    reason = "every row already has a current observation";
                    break;
                case State.InFlight:
                    reason = "an action is in flight; read it with GetAction() first";
                    break;
                default:
                    reason = "the rows have not reported what followed the last action";
                    break;
            }
            throw new InvalidOperationException($"{call} is not valid here - {reason}.");
        }

        private void RequireNotDisposed()
        {
            if (_state == State.Disposed)
            {
                throw new ObjectDisposedException(nameof(PolicyRunner));
            }
        }

        private void RequireRow(int row)
        {
            if (row < 0 || row >= BatchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is outside the batch of {BatchSize}.");
            }
        }

        private static void RequireLength(float[] values, int expected, string name)
        {
            if (values == null) throw new ArgumentNullException(name);
            if (values.Length != expected)
            {
                throw new ArgumentException($"Expected {expected} values, got {values.Length}.", name);
            }
        }
    }
}
