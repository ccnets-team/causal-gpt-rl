using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace CCNets.CausalGPTRL
{
    /// <summary>
    /// Represents one entry in `state_specs` or `action_specs`, matching
    /// `SpaceSpec.to_json_dict` in causal_gpt_rl. The `start` field is stored in the declared
    /// container instead.
    /// </summary>
    public sealed class SpaceSpec
    {
        public SpaceSpec(string type, int size, string dtype, float[] low, float[] high, string squash)
        {
            Type = type;
            Size = size;
            Dtype = dtype;
            // Copy behind a read-only view, for the same reason ActionLayout does: the
            // validator checks these bounds and the decode then clips against them, so a
            // spec that stays editable afterwards is a check that can be walked back after
            // it has passed.
            Low = new ReadOnlyCollection<float>(new List<float>(low ?? Array.Empty<float>()));
            High = new ReadOnlyCollection<float>(new List<float>(high ?? Array.Empty<float>()));
            Squash = squash;
        }

        /// <summary>"continuous", "discrete", "multi_discrete", or "multi_binary".</summary>
        public string Type { get; }
        public int Size { get; }
        public string Dtype { get; }
        /// <summary>Lower bound per column. Read-only, and a copy of what was declared.</summary>
        public IReadOnlyList<float> Low { get; }

        /// <summary>Upper bound per column.</summary>
        public IReadOnlyList<float> High { get; }

        /// <summary>"tanh" or null.</summary>
        public string Squash { get; }

        public bool IsContinuous => Type == "continuous";

        public static SpaceSpec FromJson(JObject payload)
        {
            if (payload == null) throw new ArgumentNullException(nameof(payload));

            var size = BundleJson.RequireInt(payload, "size");
            return new SpaceSpec(
                BundleJson.RequireString(payload, "type"),
                size,
                payload.Value<string>("dtype"),
                BundleJson.ReadBounds(payload["low"], size, float.NegativeInfinity),
                BundleJson.ReadBounds(payload["high"], size, float.PositiveInfinity),
                payload.Value<string>("squash"));
        }
    }

    /// <summary>
    /// Represents a declared Gymnasium space from `state_container` or `action_container`.
    /// Only the fields used by the runtime are exposed: the kind, the
    /// per-dimension `start` offsets, and nested leaves for Tuple/Dict.
    /// </summary>
    public sealed class SpaceContainer
    {
        public SpaceContainer(
            string type,
            int[] start,
            IReadOnlyList<SpaceContainer> spaces,
            IReadOnlyList<string> keys)
        {
            Type = type;
            Start = start;
            Spaces = spaces;
            Keys = keys;
        }

        /// <summary>"Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", or "Dict".</summary>
        public string Type { get; }

        /// <summary>
        /// Index offsets the environment expects added back after decoding, or null when
        /// the space declares none. Discrete emits a single value; MultiDiscrete emits one
        /// per dimension and only when some entry is non-zero.
        /// </summary>
        public int[] Start { get; }

        /// <summary>Leaves of a Tuple/Dict container, in declared order. Empty otherwise.</summary>
        public IReadOnlyList<SpaceContainer> Spaces { get; }

        /// <summary>
        /// Dict keys, positionally aligned with <see cref="Spaces"/>. Empty for every other
        /// kind. Order is what matters for packing, but the keys name the leaves a caller
        /// has to concatenate, so they are kept rather than dropped.
        /// </summary>
        public IReadOnlyList<string> Keys { get; }

        public bool IsContainer => Type == "Tuple" || Type == "Dict";

        public bool HasNonZeroStart
        {
            get
            {
                if (Start == null) return false;
                foreach (var offset in Start)
                {
                    if (offset != 0) return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Returns the first space or nested leaf that declares a non-zero start, or null if
        /// none does. Nested leaves are included because multi-agent bundles can declare
        /// spaces using nested dictionaries.
        /// </summary>
        public SpaceContainer FindNonZeroStart()
        {
            if (HasNonZeroStart)
            {
                return this;
            }
            foreach (var leaf in Spaces)
            {
                var found = leaf?.FindNonZeroStart();
                if (found != null)
                {
                    return found;
                }
            }
            return null;
        }

        /// <summary>This space and every leaf below it, in declared order.</summary>
        public IEnumerable<SpaceContainer> SelfAndLeaves()
        {
            yield return this;
            foreach (var leaf in Spaces)
            {
                if (leaf == null) continue;
                foreach (var nested in leaf.SelfAndLeaves())
                {
                    yield return nested;
                }
            }
        }

        public static SpaceContainer FromJson(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            if (!(token is JObject payload))
            {
                throw new BundleValidationException("A declared space must be a JSON object.");
            }

            var type = BundleJson.RequireString(payload, "type");

            int[] start = null;
            var startToken = payload["start"];
            if (startToken != null && startToken.Type != JTokenType.Null)
            {
                start = startToken.Type == JTokenType.Array
                    ? startToken.ToObject<int[]>()
                    : new[] { startToken.Value<int>() };
            }

            var spaces = new List<SpaceContainer>();
            var keys = new List<string>();
            if (payload["spaces"] is JArray leaves)
            {
                // A Tuple serializes its leaves directly, while a Dict serializes ordered
                // [key, subspace] pairs — an array, not an object, so that key order
                // survives the round trip. See serialize_space in inference/spaces.py.
                var isDict = type == "Dict";
                foreach (var leaf in leaves)
                {
                    if (!isDict)
                    {
                        spaces.Add(FromJson(leaf));
                        continue;
                    }

                    if (!(leaf is JArray pair) || pair.Count != 2)
                    {
                        throw new BundleValidationException(
                            "A Dict space must serialize its leaves as [key, subspace] pairs.");
                    }
                    keys.Add(pair[0].Value<string>());
                    spaces.Add(FromJson(pair[1]));
                }
            }

            return new SpaceContainer(type, start, spaces, keys);
        }
    }

    /// <summary>
    /// A parsed bundle `config.json`. This carries what the runtime needs to serve an
    /// exported graph; weights and architecture stay inside the ONNX file, so
    /// `model_config` is read only where it can be cross-checked against that graph.
    /// </summary>
    public sealed class BundleConfig
    {
        public const string DefaultBosCacheMode = "discard";

        private BundleConfig(
            int bundleFormatVersion,
            string packageVersion,
            string networkName,
            int contextLength,
            int modelContextLength,
            IReadOnlyList<SpaceSpec> stateSpecs,
            IReadOnlyList<SpaceSpec> actionSpecs,
            IReadOnlyList<string> requiresCapabilities,
            SpaceContainer stateContainer,
            SpaceContainer actionContainer,
            string bosCacheMode,
            string envId)
        {
            BundleFormatVersion = bundleFormatVersion;
            PackageVersion = packageVersion;
            NetworkName = networkName;
            ContextLength = contextLength;
            ModelContextLength = modelContextLength;
            StateSpecs = stateSpecs;
            ActionSpecs = actionSpecs;
            RequiresCapabilities = requiresCapabilities;
            StateContainer = stateContainer;
            ActionContainer = actionContainer;
            BosCacheMode = bosCacheMode;
            EnvId = envId;
        }

        public int BundleFormatVersion { get; }

        /// <summary>The causal-gpt-rl version that exported the bundle. Null before 0.1.0.</summary>
        public string PackageVersion { get; }

        public string NetworkName { get; }
        public int ContextLength { get; }

        /// <summary>`model_config.context_length`, or -1 when the bundle omits it.</summary>
        public int ModelContextLength { get; }

        public IReadOnlyList<SpaceSpec> StateSpecs { get; }
        public IReadOnlyList<SpaceSpec> ActionSpecs { get; }
        public IReadOnlyList<string> RequiresCapabilities { get; }

        /// <summary>Declared observation space, or null when the bundle predates 0.6.0.</summary>
        public SpaceContainer StateContainer { get; }

        /// <summary>Declared action space, or null when the bundle predates 0.6.0.</summary>
        public SpaceContainer ActionContainer { get; }

        /// <summary>Resolved as `serving.bos_cache_mode`, falling back to "discard".</summary>
        public string BosCacheMode { get; }

        public string EnvId { get; }

        /// <summary>Model-facing observation width, summed over the state specs.</summary>
        public int StateSize => SumSizes(StateSpecs);

        /// <summary>Model-facing action width, summed over the action specs.</summary>
        public int ActionSize => SumSizes(ActionSpecs);

        public static BundleConfig FromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("Bundle config is empty.", nameof(json));
            }

            JObject payload;
            try
            {
                payload = JObject.Parse(json);
            }
            catch (Exception error)
            {
                throw new BundleValidationException($"Bundle config is not valid JSON: {error.Message}");
            }

            var modelConfig = payload["model_config"] as JObject;
            var serving = payload["serving"] as JObject;

            return new BundleConfig(
                payload.Value<int?>("bundle_format_version") ?? 0,
                payload.Value<string>("package_version"),
                modelConfig?.Value<string>("network_name"),
                BundleJson.RequireInt(payload, "context_length"),
                modelConfig?.Value<int?>("context_length") ?? -1,
                ReadSpecs(payload, "state_specs"),
                ReadSpecs(payload, "action_specs"),
                payload["requires_capabilities"]?.ToObject<string[]>() ?? Array.Empty<string>(),
                SpaceContainer.FromJson(payload["state_container"]),
                SpaceContainer.FromJson(payload["action_container"]),
                serving?.Value<string>("bos_cache_mode") ?? DefaultBosCacheMode,
                payload.Value<string>("env_id"));
        }

        private static IReadOnlyList<SpaceSpec> ReadSpecs(JObject payload, string key)
        {
            if (!(payload[key] is JArray entries))
            {
                throw new BundleValidationException($"Bundle config is missing required key '{key}'.");
            }

            var specs = new List<SpaceSpec>(entries.Count);
            foreach (var entry in entries)
            {
                specs.Add(SpaceSpec.FromJson(entry as JObject));
            }
            return specs;
        }

        private static int SumSizes(IReadOnlyList<SpaceSpec> specs)
        {
            var total = 0;
            foreach (var spec in specs)
            {
                total += spec.Size;
            }
            return total;
        }
    }

    internal static class BundleJson
    {
        internal static string RequireString(JObject payload, string key)
        {
            var value = payload.Value<string>(key);
            if (string.IsNullOrEmpty(value))
            {
                throw new BundleValidationException($"Bundle config is missing required key '{key}'.");
            }
            return value;
        }

        internal static int RequireInt(JObject payload, string key)
        {
            var value = payload.Value<int?>(key);
            if (value == null)
            {
                throw new BundleValidationException($"Bundle config is missing required key '{key}'.");
            }
            return value.Value;
        }

        /// <summary>
        /// Reads a bounds array. Unbounded entries are JSON null in v2 bundles and the
        /// non-standard Infinity token in v1; declared spaces spell them "inf"/"-inf".
        /// A missing array means the spec is unbounded in that direction.
        /// </summary>
        internal static float[] ReadBounds(JToken token, int size, float unbounded)
        {
            var bounds = new float[size];
            if (token == null || token.Type == JTokenType.Null)
            {
                for (var index = 0; index < size; index++)
                {
                    bounds[index] = unbounded;
                }
                return bounds;
            }

            if (!(token is JArray entries))
            {
                throw new BundleValidationException("Spec bounds must be a JSON array.");
            }
            if (entries.Count != size)
            {
                throw new BundleValidationException(
                    $"Spec declares size {size} but carries {entries.Count} bounds.");
            }

            for (var index = 0; index < size; index++)
            {
                bounds[index] = ReadBound(entries[index], unbounded);
            }
            return bounds;
        }

        private static float ReadBound(JToken entry, float unbounded)
        {
            if (entry == null || entry.Type == JTokenType.Null)
            {
                return unbounded;
            }
            if (entry.Type != JTokenType.String)
            {
                return entry.Value<float>();
            }

            var text = entry.Value<string>();
            switch (text)
            {
                case "inf":
                case "Infinity":
                    return float.PositiveInfinity;
                case "-inf":
                case "-Infinity":
                    return float.NegativeInfinity;
                default:
                    if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
                    {
                        return parsed;
                    }
                    throw new BundleValidationException($"Spec bound '{text}' is not a number.");
            }
        }
    }
}
