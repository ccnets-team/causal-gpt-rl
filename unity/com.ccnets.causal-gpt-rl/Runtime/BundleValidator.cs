using System;
using System.Collections.Generic;

namespace CCNets.CausalGPTRL
{
    /// <summary>
    /// Raised when a bundle declares something this runtime cannot serve. Refusing is
    /// deliberate: a silently mis-decoded action is far more expensive to find than a
    /// load that fails at startup with a reason.
    /// </summary>
    public sealed class BundleValidationException : Exception
    {
        public BundleValidationException(string message) : base(message)
        {
        }
    }

    /// <summary>
    /// The loud refusals in `causal_gpt_rl/inference/bundle.py` that apply to THIS path,
    /// plus the limits of this runtime's own decoding.
    ///
    /// The distinction matters. bundle.py gates `load_runner`, which reads the whole config
    /// and restores declared containers with gym.unflatten. This runtime mirrors the ONNX
    /// path instead (examples/unity/evaluate_onnx.py), which takes its contract from the
    /// graph's shapes and the environment's own action spec, and returns a flat action.
    /// Porting that gate wholesale refuses bundles the reference serves.
    ///
    /// What remains is: the refusal set equals what <see cref="ActionCodec"/> can express,
    /// and anything the decode would silently get wrong — a class offset it never adds back,
    /// a graph paired with the wrong config.
    /// </summary>
    internal static class BundleValidator
    {
        private static readonly int[] SupportedBundleFormatVersions = { 1, 2 };

        /// <summary>
        /// Capabilities this runtime advertises. "hybrid_state" is listed in the narrow
        /// sense that the caller hands over an already-flattened observation vector, so a
        /// container of continuous leaves needs no structural flattening from us. It does
        /// NOT mean this runtime implements gym.flatten the way the Python runner does —
        /// leaf order remains a contract the caller keeps, which is why
        /// <see cref="ValidateStatePacking"/> refuses every non-continuous state spec.
        /// </summary>
        private static readonly HashSet<string> SupportedCapabilities = new HashSet<string>
        {
            "hybrid_state",
        };

        /// <summary>
        /// Capabilities that describe the PyTorch loader's contract and say nothing about
        /// this one. They are accepted rather than "supported": the question they answer
        /// does not arise on the ONNX path.
        ///
        /// "action_container" declares that `load_runner` must unflatten the action into a
        /// declared Dict/Tuple space. The shipping ONNX reference (examples/unity/
        /// evaluate_onnx.py) never reads the bundle's container at all — it takes the branch
        /// layout from the environment's own action spec and returns a flat action, which is
        /// what a game applies. Refusing on this capability would reject the published
        /// multi-agent bundles (DungeonEscape, SoccerTwos) that run fine through that path.
        /// What the container does decide is the class `start` offset, checked below.
        /// </summary>
        private static readonly HashSet<string> PathIrrelevantCapabilities = new HashSet<string>
        {
            "action_container",
        };

        /// <summary>
        /// Capabilities this runtime recognizes but does not implement, so the refusal
        /// names the missing feature instead of an opaque token. An entry graduates into
        /// <see cref="SupportedCapabilities"/> once its decode path ships AND a fixture
        /// exercises it.
        /// </summary>
        private static readonly Dictionary<string, string> DeferredCapabilityReasons =
            new Dictionary<string, string>
            {
                {
                    "hybrid_action",
                    "the action schedule mixes head families (e.g. continuous + discrete); " +
                    "this runtime decodes one family per bundle"
                },
            };

        /// <summary>Serialized gym space kinds this runtime can reason about.</summary>
        private static readonly HashSet<string> KnownSpaceKinds = new HashSet<string>
        {
            "Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict",
        };

        /// <summary>
        /// Serving conventions this runtime implements. "retain" is absent on purpose: the
        /// autoregressive loop clears the episode-start token from the mask after the first
        /// act, and no fixture exercises the alternative. Opening it means implementing the
        /// retained path and generating a fixture that walks it.
        /// </summary>
        private static readonly HashSet<string> SupportedBosCacheModes = new HashSet<string>
        {
            "discard",
        };

        /// <summary>
        /// Validates everything the config can answer on its own. Call
        /// <see cref="ValidateGraph"/> as well once the ONNX model is loaded, so the
        /// declared shapes and the graph's shapes are cross-checked.
        /// </summary>
        public static void Validate(BundleConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            ValidateFormatVersion(config);
            ValidateCapabilities(config);
            ValidateServing(config);
            ValidateContextLength(config);
            ValidateStatePacking(config);
            ValidateActionDecoding(config);
        }

        /// <summary>
        /// Cross-checks the declared shapes against the loaded graph. The bundle and the
        /// exported ONNX are separate files and can be paired by mistake; the window
        /// would then be built to the wrong width and every step would be silently wrong.
        /// </summary>
        public static void ValidateGraph(
            BundleConfig config,
            int batchSize,
            int contextLength,
            int stateSize,
            int actionSize)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            if (batchSize < 1)
            {
                throw new BundleValidationException($"Graph declares batch size {batchSize}; expected at least 1.");
            }
            RequireGraphMatch("context length", config.ContextLength, contextLength);
            RequireGraphMatch("state size", config.StateSize, stateSize);
            RequireGraphMatch("action size", config.ActionSize, actionSize);
        }

        private static void ValidateFormatVersion(BundleConfig config)
        {
            if (Array.IndexOf(SupportedBundleFormatVersions, config.BundleFormatVersion) >= 0)
            {
                return;
            }

            throw new BundleValidationException(
                $"Unsupported bundle_format_version={config.BundleFormatVersion}; " +
                $"this runtime supports {string.Join(", ", SupportedBundleFormatVersions)}.");
        }

        private static void ValidateCapabilities(BundleConfig config)
        {
            var missing = new List<string>();
            foreach (var capability in config.RequiresCapabilities)
            {
                if (!SupportedCapabilities.Contains(capability) &&
                    !PathIrrelevantCapabilities.Contains(capability))
                {
                    missing.Add(capability);
                }
            }
            if (missing.Count == 0)
            {
                return;
            }

            missing.Sort(StringComparer.Ordinal);
            var details = new List<string>(missing.Count);
            foreach (var capability in missing)
            {
                details.Add(DeferredCapabilityReasons.TryGetValue(capability, out var reason)
                    ? $"  - {capability}: {reason}"
                    : $"  - {capability}");
            }

            throw new BundleValidationException(
                "Bundle requires capabilities this Unity runtime does not support:\n" +
                string.Join("\n", details));
        }

        private static void ValidateServing(BundleConfig config)
        {
            if (SupportedBosCacheModes.Contains(config.BosCacheMode))
            {
                return;
            }

            throw new BundleValidationException(
                $"Unsupported bos_cache_mode='{config.BosCacheMode}'; " +
                $"this runtime supports {string.Join(", ", SupportedBosCacheModes)}.");
        }

        private static void ValidateContextLength(BundleConfig config)
        {
            if (config.ContextLength < 1)
            {
                throw new BundleValidationException(
                    $"Bundle declares context_length={config.ContextLength}; expected at least 1.");
            }
            if (config.ModelContextLength >= 0 && config.ModelContextLength != config.ContextLength)
            {
                throw new BundleValidationException(
                    $"Bundle context_length={config.ContextLength} disagrees with " +
                    $"model_config.context_length={config.ModelContextLength}.");
            }
        }

        /// <summary>
        /// The runtime consumes an already-flattened observation vector, so it can check the
        /// width but never the packing order — two Box leaves swapped keep the total size and
        /// break the model's reading of every field. An all-continuous state is therefore not
        /// verified here, only permitted under an explicit packing contract the caller keeps:
        /// concatenate the declared leaves in declared order. A discrete or binary leaf would
        /// additionally need one-hot encoding and a continuous-first reorder in game code, so
        /// such a bundle is refused rather than trusted.
        /// </summary>
        private static void ValidateStatePacking(BundleConfig config)
        {
            for (var index = 0; index < config.StateSpecs.Count; index++)
            {
                var spec = config.StateSpecs[index];
                if (spec.IsContinuous)
                {
                    continue;
                }

                throw new BundleValidationException(
                    $"State spec {index} is '{spec.Type}', but this runtime takes a pre-flattened " +
                    "observation vector and cannot verify one-hot packing or continuous-first order. " +
                    "Serve this bundle from the Python runner instead.");
            }
        }

        private static void ValidateActionDecoding(BundleConfig config)
        {
            if (config.ActionSpecs.Count == 0)
            {
                throw new BundleValidationException("Bundle declares no action specs.");
            }

            ValidateActionContainer(config.ActionContainer);

            // Head types with a decode AND a fixture that walks it. "multi_binary" is absent:
            // it needs a per-leaf Bernoulli threshold rather than an argmax, and no published
            // bundle exercises one. A type opens here only when both halves exist.
            var branchCount = 0;
            var continuousCount = 0;
            for (var index = 0; index < config.ActionSpecs.Count; index++)
            {
                var spec = config.ActionSpecs[index];
                if (spec.IsContinuous)
                {
                    continuousCount++;
                    // The decode reads continuous columns first, so a continuous head declared
                    // after a branch would take the wrong slice at the right shape.
                    if (branchCount > 0)
                    {
                        throw new BundleValidationException(
                            $"Action spec {index} is continuous but follows a branch head; " +
                            "this runtime requires continuous heads to be declared first.");
                    }
                    RequireUnitBounds(spec, index);
                    continue;
                }

                if (spec.Type != "discrete" && spec.Type != "multi_discrete")
                {
                    throw new BundleValidationException(
                        $"Action spec {index} is '{spec.Type}', which this runtime cannot decode. " +
                        "Serve this bundle from the Python runner until the matching decode ships here.");
                }
                branchCount++;
            }

            // A mixed schedule decodes on the same code path, but no bundle exercises it, so
            // it stays refused: an unverified decode that silently emits the wrong action is
            // worse than a load that fails with a reason. Bundles that need it also declare
            // the "hybrid_action" capability, which is refused above; this catches the rest.
            if (continuousCount > 0 && branchCount > 0)
            {
                throw new BundleValidationException(
                    "Action schedule mixes continuous and branch heads; this runtime decodes " +
                    "one family per bundle until a fixture exercises the mixed path.");
            }
        }

        /// <summary>
        /// A Dict or Tuple container is not a refusal. The decode returns the flat action a
        /// game applies, and the declared structure is the collection wrapper's shape, not
        /// something this path restores. Two things still have to hold: every declared kind
        /// must be one this runtime recognizes, and no leaf may carry a class offset, since
        /// the decode emits bare 0-based indices and never adds one back.
        /// </summary>
        private static void ValidateActionContainer(SpaceContainer container)
        {
            if (container == null)
            {
                return;
            }

            foreach (var space in container.SelfAndLeaves())
            {
                if (!KnownSpaceKinds.Contains(space.Type))
                {
                    throw new BundleValidationException(
                        $"Action container declares an unknown space type '{space.Type}'; " +
                        $"this runtime knows {string.Join(", ", KnownSpaceKinds)}.");
                }
            }

            var offset = container.FindNonZeroStart();
            if (offset != null)
            {
                throw new BundleValidationException(
                    $"Action container declares a non-zero start offset on a {offset.Type} leaf; " +
                    "this runtime emits 0-based indices and does not add the offset back.");
            }
        }

        /// <summary>
        /// Continuous decoding clips to [-1, 1] rather than to the declared bounds, matching
        /// the exported tanh head. Any other bound would be silently ignored.
        /// </summary>
        private static void RequireUnitBounds(SpaceSpec spec, int specIndex)
        {
            for (var index = 0; index < spec.Size; index++)
            {
                if (spec.Low[index] == -1.0f && spec.High[index] == 1.0f)
                {
                    continue;
                }

                throw new BundleValidationException(
                    $"Continuous action spec {specIndex} declares bounds " +
                    $"[{spec.Low[index]}, {spec.High[index]}] at index {index}; " +
                    "this runtime clips to [-1, 1] and would ignore them.");
            }
        }

        private static void RequireGraphMatch(string label, int declared, int graph)
        {
            if (declared == graph)
            {
                return;
            }

            throw new BundleValidationException(
                $"Bundle declares {label} {declared} but the graph expects {graph}; " +
                "the config and the ONNX file are not from the same export.");
        }
    }
}
