using System.Collections.Generic;
using CCNets.CausalGPTRL;
using Unity.InferenceEngine;
using UnityEngine;

namespace CCNets.CausalGPTRL.Samples
{
    /// <summary>
    /// Drives one policy bundle through a full step: observe, request, apply, restart the
    /// rows whose episode ended, observe again. The two halves of the action request are
    /// kept apart on purpose, so no frame blocks on the GPU. This sample polls across
    /// frames, so a decision costs at least two of them; schedule early and read back later
    /// in the same frame if you would rather have one.
    ///
    /// Everything game-specific is left as a stub: packing observations and applying an
    /// action are yours, and the packing must match the one behind your bundle.
    /// </summary>
    public sealed class QuickstartAgent : MonoBehaviour
    {
        [Header("Bundle")]
        [Tooltip("The policy graph, imported as a ModelAsset.")]
        [SerializeField] private ModelAsset policy;

        [Tooltip("The bundle's config.json, imported as a TextAsset.")]
        [SerializeField] private TextAsset config;

        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        private PolicyRunner _runner;
        private ActionRequest _request;
        private float[] _observations;
        private int[] _branches;
        private readonly List<int> _finished = new List<int>();

        private void Start()
        {
            // Load validates the bundle and pairs it with the graph. A bundle this runtime
            // cannot serve throws here, naming the reason, rather than acting wrongly later.
            _runner = PolicyRunner.Load(policy, config.text, backend);
            _observations = new float[_runner.BatchSize * _runner.StateSize];

            // Sizing this from the layout is what keeps a purely continuous policy from
            // indexing a branch that does not exist, and a MultiDiscrete one from losing
            // every branch but the first. Bundles that mix both are refused at load today.
            _branches = new int[_runner.ActionLayout.BranchSizes.Count];

            PackObservations(_observations);
            _runner.Reset(_observations);
        }

        private void Update()
        {
            if (_runner == null)
            {
                return;
            }

            if (_request == null)
            {
                // Schedules without waiting. Do this as early in the frame as you can.
                _request = _runner.RequestAction();
                return;
            }

            if (!_request.IsDone)
            {
                // Nothing to do yet. Returning here spends the wait on the rest of your
                // frame instead of blocking; call GetAction() directly if you would rather
                // take the stall than skip a tick.
                return;
            }

            var action = _request.GetAction();
            _request = null;

            _finished.Clear();
            for (var row = 0; row < _runner.BatchSize; row++)
            {
                var continuous = action.Continuous(row);
                if (_branches.Length > 0)
                {
                    action.CopyDiscrete(row, _branches);
                }

                if (ApplyAction(row, continuous, _branches))
                {
                    _finished.Add(row);
                }
            }

            // Order is the contract: apply first, then restart the rows that ended, then
            // observe. The episode-start marker attaches at the next observation.
            if (_finished.Count > 0)
            {
                _runner.ResetRows(_finished);
            }

            PackObservations(_observations);
            _runner.Observe(_observations);
        }

        private void OnDestroy()
        {
            _runner?.Dispose();
            _runner = null;
            _request = null;
        }

        /// <summary>
        /// Fill <paramref name="destination"/> with BatchSize * StateSize values, row by row.
        /// This must be the same packing the trajectories behind your bundle used — the
        /// runtime can check the size but never the order.
        /// </summary>
        private void PackObservations(float[] destination)
        {
            System.Array.Clear(destination, 0, destination.Length);
        }

        /// <summary>
        /// Apply one row's action. Return true if that row's episode ended.
        /// <paramref name="branches"/> holds one chosen index per branch, in declared order;
        /// it is empty for a purely continuous policy.
        /// </summary>
        private bool ApplyAction(int row, System.ReadOnlySpan<float> continuous, int[] branches)
        {
            return false;
        }
    }
}
