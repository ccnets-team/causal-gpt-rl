using System;
using System.Collections.Generic;

namespace CCNets.CausalGPTRL
{
    internal sealed class WindowInputs
    {
        public WindowInputs(float[] states, float[] actions, float[] isBos, float[] mask)
        {
            States = states;
            Actions = actions;
            IsBos = isBos;
            Mask = mask;
        }

        public float[] States { get; }
        public float[] Actions { get; }
        public float[] IsBos { get; }
        public float[] Mask { get; }
    }

    /// <summary>
    /// Stateless-ONNX rolling context. The extra trailing slot stages the next state;
    /// model inputs always expose the first ContextLength slots.
    /// </summary>
    internal sealed class WindowContext
    {
        private readonly int _internalLength;
        private readonly float[] _states;
        private readonly float[] _actions;
        private readonly float[] _isBos;
        private readonly float[] _mask;

        public WindowContext(int batchSize, int contextLength, int stateSize, int actionSize)
        {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (contextLength <= 0) throw new ArgumentOutOfRangeException(nameof(contextLength));
            if (stateSize <= 0) throw new ArgumentOutOfRangeException(nameof(stateSize));
            if (actionSize <= 0) throw new ArgumentOutOfRangeException(nameof(actionSize));

            BatchSize = batchSize;
            ContextLength = contextLength;
            StateSize = stateSize;
            ActionSize = actionSize;
            _internalLength = contextLength + 1;
            _states = new float[batchSize * _internalLength * stateSize];
            _actions = new float[batchSize * _internalLength * actionSize];
            _isBos = new float[batchSize * _internalLength];
            _mask = new float[batchSize * _internalLength];
            Array.Fill(_isBos, 1.0f);
        }

        public int BatchSize { get; }
        public int ContextLength { get; }
        public int StateSize { get; }
        public int ActionSize { get; }

        public void Reset(float[] initialStates)
        {
            RequireLength(initialStates, BatchSize * StateSize, nameof(initialStates));
            Array.Clear(_states, 0, _states.Length);
            Array.Clear(_actions, 0, _actions.Length);
            Array.Fill(_isBos, 1.0f);
            Array.Clear(_mask, 0, _mask.Length);
            Update(initialStates, new float[BatchSize * ActionSize], 1.0f);
        }

        public void Update(float[] nextStates, float[] feedbackActions, float isBos)
        {
            var perRow = new float[BatchSize];
            if (isBos != 0.0f)
            {
                Array.Fill(perRow, isBos);
            }
            Update(nextStates, feedbackActions, perRow);
        }

        /// <summary>
        /// Advances the window with a per-row episode-start flag. Rows restart independently
        /// in a batched scene — one agent respawning must not mark the other fifteen.
        /// </summary>
        public void Update(float[] nextStates, float[] feedbackActions, float[] isBosPerRow)
        {
            RequireLength(nextStates, BatchSize * StateSize, nameof(nextStates));
            RequireLength(feedbackActions, BatchSize * ActionSize, nameof(feedbackActions));
            RequireLength(isBosPerRow, BatchSize, nameof(isBosPerRow));

            RollLeft(_states, StateSize);
            RollLeft(_actions, ActionSize);
            RollLeft(_isBos, 1);
            RollLeft(_mask, 1);

            for (var row = 0; row < BatchSize; row++)
            {
                var isBos = isBosPerRow[row];
                var stateSource = row * StateSize;
                var stateRow = row * _internalLength * StateSize;
                Array.Copy(nextStates, stateSource, _states, stateRow + (_internalLength - 1) * StateSize, StateSize);
                if (isBos != 0.0f)
                {
                    Array.Copy(nextStates, stateSource, _states, stateRow + (_internalLength - 2) * StateSize, StateSize);
                }

                Array.Copy(
                    feedbackActions,
                    row * ActionSize,
                    _actions,
                    row * _internalLength * ActionSize + (_internalLength - 2) * ActionSize,
                    ActionSize);
                _isBos[row * _internalLength + _internalLength - 2] = isBos;
                _mask[row * _internalLength + _internalLength - 2] = 1.0f;
            }
        }

        /// <summary>
        /// Ends the episode on the given rows: their buffered trajectory is wiped, their
        /// history disowned, and they are seeded to start fresh. Every other row keeps its
        /// context. Mirrors `PolicyRunner.reset_rows`.
        ///
        /// The row's carried feedback action is NOT this type's to clear — the caller holds
        /// it, and a stale action fed back on the next step would let a despawned agent's
        /// last move steer whoever reuses the row.
        ///
        /// The rows are marked as episode starts, but that flag reaches the window on the
        /// NEXT update, not now: the reset happens when the episode ends, while BOS belongs
        /// to the observation that begins the next one.
        /// </summary>
        public void ResetRows(IReadOnlyList<int> rows)
        {
            if (rows == null) throw new ArgumentNullException(nameof(rows));

            // Check every row before clearing any: a bad index halfway through would leave
            // part of the batch reset and the caller's own bookkeeping never updated.
            foreach (var row in rows)
            {
                if (row < 0 || row >= BatchSize)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(rows), $"Row {row} is outside the batch of {BatchSize}.");
                }
            }

            foreach (var row in rows)
            {
                Array.Clear(_states, row * _internalLength * StateSize, _internalLength * StateSize);
                Array.Clear(_actions, row * _internalLength * ActionSize, _internalLength * ActionSize);
                Array.Clear(_mask, row * _internalLength, _internalLength);
                Array.Fill(_isBos, 1.0f, row * _internalLength, _internalLength);
            }
        }

        public void AfterActDiscardBos()
        {
            for (var index = 0; index < _mask.Length; index++)
            {
                if (_isBos[index] != 0.0f)
                {
                    _mask[index] = 0.0f;
                }
            }
        }

        public WindowInputs Inputs()
        {
            return new WindowInputs(
                VisibleCopy(_states, StateSize),
                VisibleCopy(_actions, ActionSize),
                VisibleCopy(_isBos, 1),
                VisibleCopy(_mask, 1));
        }

        private void RollLeft(float[] values, int width)
        {
            var rowLength = _internalLength * width;
            for (var row = 0; row < BatchSize; row++)
            {
                var offset = row * rowLength;
                Array.Copy(values, offset + width, values, offset, ContextLength * width);
            }
        }

        private float[] VisibleCopy(float[] values, int width)
        {
            var visibleRowLength = ContextLength * width;
            var internalRowLength = _internalLength * width;
            var copy = new float[BatchSize * visibleRowLength];
            for (var row = 0; row < BatchSize; row++)
            {
                Array.Copy(values, row * internalRowLength, copy, row * visibleRowLength, visibleRowLength);
            }
            return copy;
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
