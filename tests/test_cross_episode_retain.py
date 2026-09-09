"""Real-model tests for token order and independent retained row timelines."""
import json

import numpy as np
import pytest
import torch

from causal_gpt_rl.inference.bundle import load_runner, CROSS_EPISODE_CAPABILITY
from causal_gpt_rl.inference.evaluation.episode_loop import run_episodes
from collection.runner import CollectionRunner
from test_cached_context_buffer import _runner
from test_bos_cache_mode import _model, _CFG, _Norm, _ACTION_SPACE
from test_pre_tanh_rollout_context import make_bundle


def record_inputs(runner):
    tokens = []
    original = runner.model.infer_cached

    def capture(x, *args, **kwargs):
        tokens.append(x.detach().cpu().numpy().copy())
        return original(x, *args, **kwargs)

    runner.model.infer_cached = capture
    return tokens


@pytest.mark.parametrize('windowed', [False, True])
@pytest.mark.parametrize('cap', [1, 4])
def test_partial_discard_restart_refuses_without_mutation_and_advance_matches_legacy(windowed, cap):
    runner = _runner(2, windowed=windowed, kv=cap, bos='discard')
    reference = _runner(2, windowed=windowed, kv=cap, bos='discard')
    for r in (runner, reference):
        r.reset(np.zeros((2, 2), np.float32))
        for step in range(3):
            r.act()
            r.observe(np.full((2, 2), step + 1, np.float32))
        r.act()
    names = ('states', 'actions', 'is_bos', 'masks')
    context = {name: getattr(runner.buffer, name).copy() for name in names}
    feedback = runner._last_buffer_action.copy()
    flags = {name: getattr(runner, name).copy() for name in (
        '_pending_bos_mask', '_pending_bos_discard_mask', '_retain_pending_action',
        '_retain_dirty', '_retain_awaiting', '_retain_has_action',
    )}
    valid = runner.buffer.get_kv_valid_lengths().copy()
    cache = runner.buffer.get_past_key_values()
    kv = [(layer.keys.clone(), layer.values.clone()) for layer in getattr(cache, 'layers', ())]
    observations = np.array([[20, 21], [4, 5]], np.float32)
    with pytest.raises(ValueError, match='advance'):
        runner.restart_episode(observations, done_mask=[True, False])
    for name, value in context.items():
        np.testing.assert_array_equal(getattr(runner.buffer, name), value)
    for name, value in flags.items():
        np.testing.assert_array_equal(getattr(runner, name), value)
    np.testing.assert_array_equal(runner._last_buffer_action, feedback)
    np.testing.assert_array_equal(runner.buffer.get_kv_valid_lengths(), valid)
    assert runner.buffer.get_past_key_values() is cache
    for layer, (keys, values) in zip(getattr(cache, 'layers', ()), kv):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)

    runner.advance(observations, done_mask=[True, False])
    reference.reset_rows([True, False])
    reference.observe(observations)
    for name in names:
        np.testing.assert_array_equal(getattr(runner.buffer, name), getattr(reference.buffer, name))
    assert runner.buffer.is_bos[0, -2, 0] == 1
    np.testing.assert_array_equal(runner.act(), reference.act())


@pytest.mark.parametrize('windowed', [False, True])
@pytest.mark.parametrize('rows', [1, 3])
@pytest.mark.parametrize('eos', [False, True])
def test_all_paused_info_has_none_without_inference(windowed, rows, eos, monkeypatch):
    runner = _runner(rows, windowed=windowed)
    if not eos:
        runner.model.termination_index = None
    observations = np.zeros((rows, 2), np.float32)
    runner.reset(observations)
    runner.act()
    runner.finish_episode()
    valid = runner.buffer.get_kv_valid_lengths().copy()
    states = runner.buffer.states.copy()
    feedback = runner._last_buffer_action.copy()

    def unexpected_forward(*args, **kwargs):
        pytest.fail('Paused rows must not run model inference')

    with monkeypatch.context() as patch:
        patch.setattr(runner.model.backbone, 'infer', unexpected_forward)
        patch.setattr(runner.model.backbone, 'forward', unexpected_forward)
        action, info = runner.act_with_info()
        assert info == {'termination_prob': None}
        assert np.asarray(action).shape == ((2,) if rows == 1 else (rows, 2))
        np.testing.assert_array_equal(runner.act(), action)
    np.testing.assert_array_equal(runner.buffer.states, states)
    np.testing.assert_array_equal(runner._last_buffer_action, feedback)
    np.testing.assert_array_equal(runner.buffer.get_kv_valid_lengths(), valid)
    runner.restart_episode(observations)
    _, resumed = runner.act_with_info()
    assert (resumed['termination_prob'] is None) == (not eos)


@pytest.mark.parametrize("cap", [1, 3, 32])
@pytest.mark.parametrize("observed", [False, True])
def test_terminal_and_bos_are_ingested_once(cap, observed):
    runner = _runner(1, ctx=8, kv=cap)
    inputs = record_inputs(runner)
    runner.reset([1, 2])
    runner.act()
    feedback = runner._last_buffer_action.copy()
    if observed:
        runner.observe([8, 9])
    runner.restart_episode([20, 21], terminal_state=[8, 9])
    assert len(inputs) == 2  # BOS, terminal action (no action generated by restart)
    np.testing.assert_array_equal(inputs[-1][0, 0, :2], [1, 2])
    np.testing.assert_array_equal(inputs[-1][0, 0, 2:4], feedback[0])
    runner.act()
    np.testing.assert_array_equal(inputs[-1][0, 0], [20, 21, 0, 0, 1])
    runner.observe([22, 23])
    runner.act()
    np.testing.assert_array_equal(inputs[-1][0, 0, :2], [20, 21])
    assert inputs[-1][0, 0, -1] == 0
    assert runner.buffer.get_kv_cache_length() <= cap


@pytest.mark.parametrize("network", ["Llama", "GPT2"])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("cap", [1, 4, 32])
def test_mixed_rows_match_independent_sessions(network, windowed, cap):
    batch = _runner(3, ctx=8, kv=cap, windowed=windowed, network=network)
    singles = [_runner(1, ctx=8, kv=cap, windowed=windowed, network=network) for _ in range(3)]
    rng = np.random.default_rng(44)
    obs = rng.normal(size=(3, 2)).astype(np.float32)
    batch.reset(obs)
    for row, runner in enumerate(singles):
        runner.reset(obs[row])
    for step in range(12):
        actual = batch.act()
        expected = np.stack([runner.act() for runner in singles])
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        obs = rng.normal(size=(3, 2)).astype(np.float32)
        done = np.array([step % 2 == 0, step % 3 == 0, step % 5 == 0])
        batch.advance(obs, done_mask=done)
        for row, runner in enumerate(singles):
            if done[row]:
                runner.restart_episode(obs[row])
            else:
                runner.observe(obs[row])
        if not windowed:
            np.testing.assert_array_equal(batch.buffer.get_kv_valid_lengths(), [r.buffer.get_kv_valid_lengths()[0] for r in singles])


@pytest.mark.parametrize("windowed", [False, True])
def test_next_step_pauses_ended_rows_and_preserves_episode_files(tmp_path, windowed):
    runner = _runner(2, ctx=16, kv=16, windowed=windowed)
    runner.action_space = _ACTION_SPACE
    collector = CollectionRunner(runner, tmp_path)
    tokens = record_inputs(runner) if not windowed else None
    collector.reset(np.zeros((2, 2), np.float32))
    first = collector.act().copy()
    collector.observe(np.ones((2, 2), np.float32), [1, 2], [True, False], [False, False])
    before = runner.buffer.states[0].copy()
    length = runner.buffer.get_kv_valid_lengths()[0]
    collector.act()  # row 0 action is ignored by NEXT_STEP
    np.testing.assert_array_equal(runner.buffer.states[0], before)
    assert runner.buffer.get_kv_valid_lengths()[0] == length
    collector.observe(np.full((2, 2), 2, np.float32), [0, 3], [False, False], [False, False])
    last = collector.act().copy()
    collector.observe(np.full((2, 2), 3, np.float32), [4, 5], [True, True], [False, False])
    collector.close()
    episodes = [np.load(path) for path in sorted(tmp_path.glob('ep_*.npz'))]
    assert sorted(len(ep['actions']) for ep in episodes) == [1, 1, 3]
    np.testing.assert_array_equal(episodes[0]['actions'][0], first[0])
    np.testing.assert_array_equal(episodes[1]['actions'][0], last[0])
    if tokens is not None:
        # Per-row cached token totals: row 0 BOS,a,BOS,a; row 1 BOS,a,a,a.
        assert sum(x.shape[0] * x.shape[1] for x in tokens) == 8


@pytest.mark.parametrize("windowed", [False, True])
def test_direct_terminal_feedback_survives_saturation(tmp_path, windowed):
    make_bundle(tmp_path, saturated=True)
    runner = load_runner(tmp_path, bos_cache_mode="retain", use_windowed=windowed)
    tokens = record_inputs(runner) if not windowed else None
    runner.reset(np.zeros(3, np.float32))
    action = runner.act()
    original = runner._last_buffer_action.copy()
    action[:] = 99
    runner.restart_episode(np.ones(3, np.float32))
    if windowed:
        np.testing.assert_array_equal(runner.buffer.actions[0, -3], original[0])
    else:
        np.testing.assert_array_equal(tokens[-1][0, 0, 3:-1], original[0])
    np.testing.assert_array_equal(runner.buffer.actions[0, -2], 0)


def test_finish_restart_errors_and_explicit_reset():
    runner = _runner(1)
    runner.reset([0, 0])
    with pytest.raises(RuntimeError):
        runner.restart_episode([1, 1])
    runner.act()
    runner.finish_episode()
    with pytest.raises(RuntimeError):
        runner.finish_episode()
    runner.restart_episode([1, 1])
    with pytest.raises(RuntimeError):
        runner.restart_episode([2, 2])
    runner.reset([3, 3])
    assert runner.buffer.get_kv_cache_length() == 0
    runner.act()
    assert runner.buffer.get_kv_cache_length() == 1


def test_new_rows_and_session_row_reset_after_mixed_restart():
    runner = _runner(2)
    runner.reset(np.zeros((2, 2), np.float32))
    runner.act()
    runner.advance(np.ones((2, 2), np.float32), done_mask=[True, False])
    runner.act()
    runner.add_rows([[5, 5]])
    runner.observe(np.full((3, 2), 6, np.float32))
    runner.act()
    assert runner.buffer.get_kv_valid_lengths()[2] == 1
    runner.reset_rows([False, True, False])
    assert runner.buffer.get_kv_valid_lengths()[1] == 0
    runner.observe(np.full((3, 2), 7, np.float32))
    runner.act()
    assert runner.buffer.get_kv_valid_lengths()[1] == 1


@pytest.mark.parametrize("ending", ["terminated", "truncated", "max_steps"])
def test_evaluation_restarts_without_merging_statistics(ending):
    class Env:
        def reset(self, **kwargs):
            self.steps = 0
            return np.zeros(2, np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.full(2, self.steps, np.float32), 2., ending == "terminated", ending == "truncated", {}

    runner = _runner(1, kv=32)
    inputs = record_inputs(runner)
    result = run_episodes(Env(), runner, num_episodes=3, max_steps=1)
    assert result['returns'] == [2., 2., 2.]
    assert result['lengths'] == [1, 1, 1]
    assert [int(x[0, 0, -1]) for x in inputs] == [1, 0, 1, 0, 1, 0]


def test_retain_metadata_is_generated_and_required(tmp_path):
    from causal_gpt_rl.inference.bundle import export_bundle
    model = _model()
    export_bundle(tmp_path, model=model, model_config=_CFG, state_specs=model.state_specs,
                  action_specs=model.action_specs, context_length=8, action_space=_ACTION_SPACE,
                  state_normalizer=_Norm(2), bos_cache_mode='retain')
    config_path = tmp_path / 'config.json'
    config = json.loads(config_path.read_text())
    assert CROSS_EPISODE_CAPABILITY in config['requires_capabilities']
    assert config['serving'] == {'bos_cache_mode': 'retain'}
    assert load_runner(tmp_path).bos_cache_mode == 'retain'
    assert load_runner(tmp_path, bos_cache_mode='discard').bos_cache_mode == 'discard'
    config['requires_capabilities'].remove(CROSS_EPISODE_CAPABILITY)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='re-export'):
        load_runner(tmp_path)


def test_replacement_requires_session_reset():
    runner = _runner(1)
    runner.reset([0, 0])
    runner.act()
    runner.model = _runner(1).model
    with pytest.raises(RuntimeError, match='replacement'):
        runner.restart_episode([1, 1])
    with pytest.raises(RuntimeError, match='replacement'):
        runner.act()
    runner.reset([1, 1])
    runner.act()


@pytest.mark.parametrize('declared', [False, True])
def test_checkpoint_retain_mode_and_feedback(tmp_path, declared):
    from causal_gpt_rl.inference.runner import PolicyRunner
    from test_action_normalization import components
    model, _, _, specs, _ = components()
    payload = {'model_state': model.state_dict(),
               'rollout_context': {'action_coordinate': 'standardized_pre_tanh_v1'}}
    if declared:
        payload.update(serving={'bos_cache_mode': 'retain'},
                       requires_capabilities=[CROSS_EPISODE_CAPABILITY])
    path = tmp_path / 'checkpoint.pt'
    torch.save(payload, path)
    runner = PolicyRunner.from_checkpoint(
        path, model=components()[0], action_specs=specs, state_size=3,
        context_length=4, bos_cache_mode=None if declared else 'retain',
    )
    assert runner.bos_cache_mode == 'retain'
    runner.reset([0, 0, 0])
    runner.act()
    runner.restart_episode([1, 1, 1])
    runner.act()
    assert runner.buffer.get_kv_cache_length() == 3


def test_single_collection_example_selects_retain(tmp_path):
    from examples.deploy.record import record_episodes

    class Env:
        def reset(self, **kwargs):
            return np.zeros(2, np.float32), {}

        def step(self, action):
            return np.ones(2, np.float32), 1., True, False, {}

    runner = _runner(1, kv=32)
    runner.action_space = _ACTION_SPACE
    inputs = record_inputs(runner)
    collector = CollectionRunner(runner, tmp_path)
    record_episodes(Env(), collector, episodes=3, max_steps=1, seed_start=0)
    collector.close()
    assert len(list(tmp_path.glob('ep_*.npz'))) == 3
    assert [int(x[0, 0, -1]) for x in inputs] == [1, 0, 1, 0, 1, 0]


def test_onnx_retain_metadata_is_preserved_and_unsupported_host_refuses(tmp_path):
    from causal_gpt_rl.inference.bundle import export_bundle
    from causal_gpt_rl.export import export_onnx
    from test_unity_onnx_bos_window import _window_class
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    model = _model()
    bundle_path = tmp_path / 'bundle'
    export_bundle(bundle_path, model=model, model_config=_CFG,
                  state_specs=model.state_specs, action_specs=model.action_specs,
                  context_length=4, action_space=_ACTION_SPACE,
                  state_normalizer=_Norm(2), bos_cache_mode='retain')
    output = tmp_path / 'model.onnx'
    export_onnx(bundle_path, output, verify=False)
    metadata = {prop.key: prop.value for prop in onnx.load(output).metadata_props}
    assert metadata['causal_gpt_rl.bos_cache_mode'] == 'retain'
    assert CROSS_EPISODE_CAPABILITY in json.loads(metadata['causal_gpt_rl.requires_capabilities'])
    # This host must reject before running the graph or touching observations.
    host_globals = _window_class().__init__.__globals__
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    with pytest.raises(ValueError, match='cross-episode retain'):
        host_globals['_run_onnx_with_context'](session, {}, 1)
