using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace CCNets.CausalGPTRL
{
    /// <summary>
    /// How a raw model output splits into heads. Continuous columns come first, then one
    /// branch per discrete head — the order the exported graph emits and the order
    /// `evaluate_onnx._decode` reads.
    /// </summary>
    public sealed class ActionLayout
    {
        /// <summary>
        /// Internal: a layout is derived from a validated bundle, never supplied. Handing the
        /// constructor out would let a caller declare bounds the validator already refused.
        /// </summary>
        internal ActionLayout(int continuousSize, IReadOnlyList<int> branches, float[] low, float[] high)
        {
            if (continuousSize < 0) throw new ArgumentOutOfRangeException(nameof(continuousSize));
            if (branches == null) throw new ArgumentNullException(nameof(branches));
            if (continuousSize > 0)
            {
                if (low == null) throw new ArgumentNullException(nameof(low));
                if (high == null) throw new ArgumentNullException(nameof(high));
                if (low.Length != continuousSize || high.Length != continuousSize)
                {
                    throw new ArgumentException(
                        $"Continuous bounds must carry {continuousSize} entries.", nameof(low));
                }
            }
            foreach (var branch in branches)
            {
                if (branch < 1)
                {
                    throw new ArgumentException($"Branch size {branch} is not positive.", nameof(branches));
                }
            }

            // Copy: the arrays come from the caller, and the decode clips against these
            // bounds on every step. A layout that can be edited afterwards is a validator that
            // can be edited afterwards.
            ContinuousSize = continuousSize;
            BranchSizes = new ReadOnlyCollection<int>(new List<int>(branches));
            Low = new ReadOnlyCollection<float>(new List<float>(low ?? Array.Empty<float>()));
            High = new ReadOnlyCollection<float>(new List<float>(high ?? Array.Empty<float>()));
        }

        public int ContinuousSize { get; }

        /// <summary>
        /// Class count per discrete head, in declared order. Sizes, not the heads themselves —
        /// `[3, 3, 3]` is three heads of three classes each.
        /// </summary>
        public IReadOnlyList<int> BranchSizes { get; }

        /// <summary>Lower bound per continuous column. Read-only, and a copy of what was declared.</summary>
        public IReadOnlyList<float> Low { get; }

        /// <summary>Upper bound per continuous column.</summary>
        public IReadOnlyList<float> High { get; }

        /// <summary>
        /// The model's action width: continuous columns plus every branch's logits. Same
        /// quantity the Python runner calls `action_size`.
        /// </summary>
        public int ActionSize
        {
            get
            {
                var width = ContinuousSize;
                foreach (var branch in BranchSizes) width += branch;
                return width;
            }
        }

        /// <summary>
        /// The environment action width: continuous columns plus ONE index per branch.
        /// Narrower than <see cref="ActionSize"/> whenever the policy has discrete heads.
        /// The Python runner has no attribute for it; `evaluate_onnx` computes it inline.
        /// </summary>
        public int EnvironmentActionSize => ContinuousSize + BranchSizes.Count;

        /// <summary>
        /// Derives the layout from the bundle's action specs. The declared container is not
        /// consulted: branch sizes are the head sizes, and this runtime emits 0-based indices
        /// (see BundleValidator, which refuses a container declaring an offset).
        /// </summary>
        public static ActionLayout FromConfig(BundleConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            var continuousSize = 0;
            var branches = new List<int>();
            var low = new List<float>();
            var high = new List<float>();

            foreach (var spec in config.ActionSpecs)
            {
                if (spec.IsContinuous)
                {
                    if (branches.Count > 0)
                    {
                        throw new BundleValidationException(
                            "A continuous action head follows a branch head; the decode reads " +
                            "continuous columns first and would take the wrong slice.");
                    }
                    continuousSize += spec.Size;
                    low.AddRange(spec.Low);
                    high.AddRange(spec.High);
                    continue;
                }

                branches.Add(spec.Size);
            }

            return new ActionLayout(continuousSize, branches, low.ToArray(), high.ToArray());
        }
    }

    /// <summary>
    /// One decode: what the environment is given, and what the window is fed back. They are
    /// different values, and conflating them breaks the rollout — continuous is clipped for
    /// the environment but fed back raw, and a branch becomes an index for the environment
    /// but a one-hot row for the window.
    /// </summary>
    public sealed class DecodedAction
    {
        public DecodedAction(float[] environmentAction, float[] feedbackAction, ActionLayout layout, int batchSize)
        {
            EnvironmentAction = environmentAction;
            FeedbackAction = feedbackAction;
            Layout = layout;
            BatchSize = batchSize;
        }

        /// <summary>
        /// Row-major, <see cref="ActionLayout.EnvironmentActionSize"/> per row. Branch results
        /// sit here as whole numbers in float columns — the shape the Python reference hands
        /// the environment, and what the fixtures record. Game code should prefer
        /// <see cref="Continuous"/> and <see cref="Discrete"/>.
        /// </summary>
        public float[] EnvironmentAction { get; }

        /// <summary>Row-major, <see cref="ActionLayout.ActionSize"/> per row.</summary>
        public float[] FeedbackAction { get; }

        public ActionLayout Layout { get; }
        public int BatchSize { get; }

        /// <summary>One row's continuous columns. A view, not a copy.</summary>
        public ReadOnlySpan<float> Continuous(int row)
        {
            RequireRow(row);
            return new ReadOnlySpan<float>(
                EnvironmentAction, row * Layout.EnvironmentActionSize, Layout.ContinuousSize);
        }

        /// <summary>
        /// One branch's chosen class for one row, as the integer a game actually applies.
        /// Allocation-free, because this is read every frame for every agent.
        /// </summary>
        public int Discrete(int row, int branch)
        {
            RequireRow(row);
            if (branch < 0 || branch >= Layout.BranchSizes.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(branch), $"Branch {branch} is outside the {Layout.BranchSizes.Count} declared heads.");
            }
            return (int)EnvironmentAction[row * Layout.EnvironmentActionSize + Layout.ContinuousSize + branch];
        }

        /// <summary>Every branch for one row, into a caller-owned buffer.</summary>
        public void CopyDiscrete(int row, int[] destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (destination.Length < Layout.BranchSizes.Count)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} entries; {Layout.BranchSizes.Count} branches were decoded.",
                    nameof(destination));
            }
            for (var branch = 0; branch < Layout.BranchSizes.Count; branch++)
            {
                destination[branch] = Discrete(row, branch);
            }
        }

        private void RequireRow(int row)
        {
            if (row < 0 || row >= BatchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is outside the batch of {BatchSize}.");
            }
        }
    }

    /// <summary>
    /// Port of `evaluate_onnx._decode`. Continuous columns are clipped to [-1, 1] — the
    /// exported head is a tanh, and BundleValidator refuses any bundle declaring other
    /// bounds rather than letting the clip silently ignore them.
    /// </summary>
    internal static class ActionCodec
    {
        public static DecodedAction Decode(float[] raw, int batchSize, ActionLayout layout)
        {
            if (raw == null) throw new ArgumentNullException(nameof(raw));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize));

            var actionSize = layout.ActionSize;
            if (raw.Length != batchSize * actionSize)
            {
                throw new ArgumentException(
                    $"Raw action carries {raw.Length} values; expected {batchSize} x {actionSize}.",
                    nameof(raw));
            }

            var environmentSize = layout.EnvironmentActionSize;
            var environment = new float[batchSize * environmentSize];
            var feedback = new float[batchSize * actionSize];

            for (var row = 0; row < batchSize; row++)
            {
                var source = row * actionSize;
                var environmentCursor = row * environmentSize;
                var feedbackCursor = row * actionSize;

                for (var index = 0; index < layout.ContinuousSize; index++)
                {
                    var value = raw[source + index];
                    environment[environmentCursor + index] =
                        Math.Max(layout.Low[index], Math.Min(layout.High[index], value));
                    feedback[feedbackCursor + index] = value;
                }

                var offset = layout.ContinuousSize;
                environmentCursor += layout.ContinuousSize;
                for (var branch = 0; branch < layout.BranchSizes.Count; branch++)
                {
                    var size = layout.BranchSizes[branch];
                    var best = 0;
                    var bestLogit = raw[source + offset];
                    for (var index = 1; index < size; index++)
                    {
                        var logit = raw[source + offset + index];
                        if (logit > bestLogit)
                        {
                            bestLogit = logit;
                            best = index;
                        }
                    }

                    // The environment gets the class index; the window gets it one-hot, so the
                    // next step reads the same width the model was trained on.
                    environment[environmentCursor + branch] = best;
                    feedback[feedbackCursor + offset + best] = 1.0f;
                    offset += size;
                }
            }

            return new DecodedAction(environment, feedback, layout, batchSize);
        }
    }
}
