using System.Runtime.CompilerServices;

// The conformance tests drive the internals directly — the window, the codec, the backend
// and the validator each have their own fixture-backed suite. They are internal because the
// supported surface is PolicyRunner; widening one later is additive, narrowing it is not.
[assembly: InternalsVisibleTo("CCNets.CausalGPTRL.Tests")]

// The performance harness measures raw schedule/readback cost, which the runner cannot
// report. It is a development consumer, not a customer of the API, so it gets the internals
// through its own assembly rather than by widening the package's public surface. Never grant
// this to "Assembly-CSharp": that is every customer's default script assembly.
[assembly: InternalsVisibleTo("CGRLTests.Performance")]
