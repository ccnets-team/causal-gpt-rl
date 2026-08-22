using System;
using NUnit.Framework;
using Newtonsoft.Json.Linq;

namespace CCNets.CausalGPTRL.Tests
{
    /// <summary>
    /// The shipped fixture config has to load, and every bundle this runtime cannot decode
    /// has to be refused with a reason. A refusal that silently disappears is the failure
    /// mode these tests exist to catch, so each case asserts the refusal happens at all —
    /// not the exact wording.
    /// </summary>
    public sealed class BundleValidatorTests
    {
        /// <summary>
        /// The refusal cases mutate one known-good config, so they are written against the
        /// Crawler bundle by name. <see cref="EveryStagedBundleValidates"/> covers the rest.
        /// </summary>
        private static string FixtureJson()
        {
            return FixtureModels.LoadText(FixtureModels.Crawler, "config.json");
        }

        private static BundleConfig FixtureConfig()
        {
            return BundleConfig.FromJson(FixtureJson());
        }

        private static void AssertRefused(string because, Action<JObject> mutate)
        {
            var root = JObject.Parse(FixtureJson());
            mutate(root);
            Assert.That(
                () => BundleValidator.Validate(BundleConfig.FromJson(root.ToString())),
                Throws.TypeOf<BundleValidationException>(),
                because);
        }

        [TestCaseSource(typeof(FixtureModels), nameof(FixtureModels.Models))]
        public void EveryStagedBundleValidates(string model)
        {
            // A fixture this runtime would refuse to serve has no business being staged:
            // the parity tests would then verify a graph the product cannot load.
            var config = BundleConfig.FromJson(FixtureModels.LoadText(model, "config.json"));

            Assert.DoesNotThrow(() => BundleValidator.Validate(config));
        }

        [Test]
        public void ShippedFixtureParses()
        {
            var config = FixtureConfig();

            Assert.That(config.BundleFormatVersion, Is.EqualTo(2));
            Assert.That(config.ContextLength, Is.EqualTo(32));
            Assert.That(config.ModelContextLength, Is.EqualTo(32));
            Assert.That(config.StateSize, Is.EqualTo(158));
            Assert.That(config.ActionSize, Is.EqualTo(20));
            Assert.That(config.BosCacheMode, Is.EqualTo("discard"));
            Assert.That(config.EnvId, Is.EqualTo("Crawler"));
            Assert.That(config.ActionContainer.Type, Is.EqualTo("Box"));
            Assert.That(config.ActionContainer.HasNonZeroStart, Is.False);
            Assert.That(config.StateContainer.Type, Is.EqualTo("Tuple"));
            Assert.That(config.StateContainer.Spaces.Count, Is.EqualTo(2));
        }

        [Test]
        public void ShippedFixtureValidates()
        {
            var config = FixtureConfig();

            Assert.DoesNotThrow(() => BundleValidator.Validate(config));
            Assert.DoesNotThrow(() => BundleValidator.ValidateGraph(config, 1, 32, 158, 20));
        }

        [Test]
        public void ParsesDictContainerPairs()
        {
            // A Dict serializes as ordered [key, subspace] pairs rather than an object, so
            // that key order survives. Reading it as an object loses the leaves entirely.
            var root = JObject.Parse(FixtureJson());
            var leaves = (JArray)root["state_container"]["spaces"];
            root["state_container"] = new JObject
            {
                ["type"] = "Dict",
                ["spaces"] = new JArray(
                    new JArray("proprioception", leaves[0].DeepClone()),
                    new JArray("target", leaves[1].DeepClone())),
            };

            var config = BundleConfig.FromJson(root.ToString());

            Assert.That(config.StateContainer.Type, Is.EqualTo("Dict"));
            Assert.That(config.StateContainer.Spaces.Count, Is.EqualTo(2));
            Assert.That(config.StateContainer.Keys, Is.EqualTo(new[] { "proprioception", "target" }));
            Assert.DoesNotThrow(() => BundleValidator.Validate(config));
        }

        [Test]
        public void RefusesMalformedDictPairs()
        {
            AssertRefused("a Dict leaf that is not a [key, subspace] pair is malformed", root =>
            {
                root["state_container"] = new JObject
                {
                    ["type"] = "Dict",
                    ["spaces"] = new JArray(new JObject { ["type"] = "Box" }),
                };
            });
        }

        [Test]
        public void RefusesUnsupportedFormatVersion()
        {
            AssertRefused("a newer on-disk layout may move fields this runtime reads",
                root => root["bundle_format_version"] = 3);
        }

        [Test]
        public void RefusesMissingRequiredKey()
        {
            AssertRefused("context_length sizes the window and has no safe default",
                root => root.Remove("context_length"));
        }

        [Test]
        public void RefusesContextLengthDisagreement()
        {
            AssertRefused("the window would be sized against one of two disagreeing lengths",
                root => root["context_length"] = 16);
        }

        [TestCase("hybrid_action", TestName = "RefusesCapability_HybridAction")]
        [TestCase("time_axis", TestName = "RefusesCapability_Unknown")]
        public void RefusesUnimplementedCapability(string capability)
        {
            AssertRefused($"{capability} is not implemented here",
                root => root["requires_capabilities"] = new JArray(capability));
        }

        [TestCase("retain", TestName = "RefusesBosCacheMode_Retain")]
        [TestCase("keep", TestName = "RefusesBosCacheMode_Unknown")]
        public void RefusesUnsupportedBosCacheMode(string mode)
        {
            AssertRefused($"the loop only implements discard, not {mode}",
                root => root["serving"] = new JObject { ["bos_cache_mode"] = mode });
        }

        [TestCase("discrete", TestName = "RefusesStateSpec_Discrete")]
        [TestCase("multi_binary", TestName = "RefusesStateSpec_MultiBinary")]
        public void RefusesNonContinuousStateSpec(string type)
        {
            AssertRefused("one-hot packing and continuous-first order cannot be verified from a size",
                root => root["state_specs"][0]["type"] = type);
        }

        [TestCase("multi_binary", TestName = "RefusesActionSpec_MultiBinary")]
        [TestCase("something_new", TestName = "RefusesActionSpec_Unknown")]
        public void RefusesUndecodableActionSpec(string type)
        {
            // multi_binary needs a per-leaf Bernoulli threshold rather than an argmax, and no
            // published bundle exercises one. A lone branch schedule declares no capability at
            // all, so the capability gate never sees these — the specs are read directly.
            AssertRefused($"ActionCodec cannot decode a '{type}' head",
                root => root["action_specs"][0]["type"] = type);
        }

        [TestCase("discrete", TestName = "AcceptsActionSpec_Discrete")]
        [TestCase("multi_discrete", TestName = "AcceptsActionSpec_MultiDiscrete")]
        public void AcceptsBranchActionSpec(string type)
        {
            // Opened once the decode shipped AND fixtures walked it (pyramids, soccertwos).
            var root = JObject.Parse(FixtureJson());
            root["action_specs"][0]["type"] = type;

            Assert.DoesNotThrow(() => BundleValidator.Validate(BundleConfig.FromJson(root.ToString())));
        }

        [Test]
        public void RefusesMixedScheduleInDecodableOrder()
        {
            // Continuous first, so the ordering rule does not catch it. The mixed path
            // decodes on the same code, but no fixture walks it.
            AssertRefused("a mixed schedule has no fixture behind it", root =>
            {
                var continuous = root["action_specs"][0].DeepClone();
                root["action_specs"] = new JArray(
                    continuous,
                    new JObject
                    {
                        ["type"] = "discrete",
                        ["size"] = 3,
                        ["dtype"] = "int64",
                        ["squash"] = null,
                    });
            });
        }

        [Test]
        public void RefusesMixedActionSchedule()
        {
            // Named for what it checks today: the branch head alone is enough to refuse.
            // Once branch decoding ships, this case has to keep failing for the ordering
            // reason instead — continuous heads must be declared before any branch.
            AssertRefused("a mixed schedule is refused while only continuous decoding exists", root =>
            {
                var continuous = root["action_specs"][0].DeepClone();
                root["action_specs"] = new JArray(
                    new JObject
                    {
                        ["type"] = "discrete",
                        ["size"] = 3,
                        ["dtype"] = "int64",
                        ["squash"] = null,
                    },
                    continuous);
            });
        }

        [Test]
        public void RefusesBoundsBeyondUnitInterval()
        {
            AssertRefused("the decode clips to [-1, 1] and would ignore a wider bound",
                root => root["action_specs"][0]["low"][0] = -2.0);
        }

        [Test]
        public void AcceptsActionContainerCapability()
        {
            // The capability gates load_runner's unflatten, not this path. Refusing on it
            // rejects the published multi-agent bundles, which the ONNX reference serves.
            var root = JObject.Parse(FixtureJson());
            root["requires_capabilities"] = new JArray("hybrid_state", "action_container");

            Assert.DoesNotThrow(() => BundleValidator.Validate(BundleConfig.FromJson(root.ToString())));
        }

        [Test]
        public void AcceptsNestedDictActionContainer()
        {
            // How DungeonEscape and SoccerTwos declare their action spaces. The decode
            // returns the flat action a game applies; the declared nesting is the
            // collection wrapper's shape and is not restored here.
            var root = JObject.Parse(FixtureJson());
            root["action_container"] = MultiAgentContainer(0);

            var config = BundleConfig.FromJson(root.ToString());

            Assert.That(config.ActionContainer.Type, Is.EqualTo("Dict"));
            Assert.That(config.ActionContainer.Keys, Is.EqualTo(new[] { "agents" }));
            Assert.DoesNotThrow(() => BundleValidator.Validate(config));
        }

        [Test]
        public void RefusesStartOffsetNestedInContainer()
        {
            // Two levels down, and still the difference between the right action and a
            // wrong one of the right shape.
            AssertRefused("a buried class offset is never added back by the decode",
                root => root["action_container"] = MultiAgentContainer(2));
        }

        [TestCase("Graph", TestName = "RefusesContainerKind_Unknown")]
        [TestCase("Sequence", TestName = "RefusesContainerKind_Sequence")]
        public void RefusesUnknownContainerKind(string kind)
        {
            AssertRefused($"'{kind}' is not a space kind this runtime can reason about",
                root => root["action_container"] = new JObject { ["type"] = kind });
        }

        [Test]
        public void RefusesMalformedContainerPairs()
        {
            AssertRefused("a Dict leaf that is not a [key, subspace] pair is malformed",
                root => root["action_container"] = new JObject
                {
                    ["type"] = "Dict",
                    ["spaces"] = new JArray(new JObject { ["type"] = "Discrete", ["n"] = 3 }),
                });
        }

        /// <summary>`{"agents": {"agent_0": MultiDiscrete([3,3,3], start)}}`.</summary>
        private static JObject MultiAgentContainer(int start)
        {
            // `new JArray(x)` with a single enumerable argument FLATTENS it, which would
            // turn a [key, subspace] pair into two loose leaves. Add the pair explicitly.
            var agent = new JObject
            {
                ["type"] = "MultiDiscrete",
                ["nvec"] = new JArray(3, 3, 3),
                ["start"] = new JArray(0, start, 0),
            };
            var agents = new JObject { ["type"] = "Dict", ["spaces"] = new JArray() };
            ((JArray)agents["spaces"]).Add(new JArray("agent_0", agent));

            var container = new JObject { ["type"] = "Dict", ["spaces"] = new JArray() };
            ((JArray)container["spaces"]).Add(new JArray("agents", agents));
            return container;
        }

        [Test]
        public void RefusesDiscreteStartOffset()
        {
            AssertRefused("a 0-based index would be emitted where the env expects an offset", root =>
            {
                root["action_container"] = new JObject
                {
                    ["type"] = "Discrete",
                    ["n"] = 3,
                    ["start"] = 1,
                };
            });
        }

        [Test]
        public void RefusesMultiDiscreteStartOffsets()
        {
            AssertRefused("per-dimension offsets are dropped by the decode", root =>
            {
                root["action_container"] = new JObject
                {
                    ["type"] = "MultiDiscrete",
                    ["nvec"] = new JArray(3, 3, 3),
                    ["start"] = new JArray(0, 2, 0),
                };
            });
        }

        [TestCase(1, 16, 158, 20, TestName = "RefusesGraph_ContextLength")]
        [TestCase(1, 32, 126, 20, TestName = "RefusesGraph_StateSize")]
        [TestCase(1, 32, 158, 3, TestName = "RefusesGraph_ActionSize")]
        [TestCase(0, 32, 158, 20, TestName = "RefusesGraph_BatchSize")]
        public void RefusesGraphMismatch(int batchSize, int contextLength, int stateSize, int actionSize)
        {
            var config = FixtureConfig();

            Assert.That(
                () => BundleValidator.ValidateGraph(config, batchSize, contextLength, stateSize, actionSize),
                Throws.TypeOf<BundleValidationException>(),
                "a config paired with the wrong graph would size the window wrongly");
        }
    }
}
