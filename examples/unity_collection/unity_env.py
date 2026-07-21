"""ML-Agents -> gymnasium wrapper.

Handles the four things a naive collection loop gets wrong: DecisionSteps/
TerminalSteps split, terminated-vs-truncated via `interrupted`, per-`agent_id`
demux across a multi-agent scene, and multi-channel observation ordering. A
termination that lands on an action-repeat gap (a step with empty
`decision_steps`) is processed rather than skipped, so episodes are not silently
merged. Runs against mlagents_envs 1.1.0.
"""
import numpy as np
from gymnasium import Env
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityEnv(Env):
    # Class-level attribute to track instance count
    _instance_count = 0
    env_dir = None
    env_id = None

    @classmethod
    def register(cls, env_id, env_entry_point, env_dir):
        """
        Register a Unity environment by setting the class-level entry_point and env_id.
        If the environment is already registered with the same values, skip updating.
        Otherwise, update the registration with the new values.
        """
        if cls.env_dir:
            if cls.env_dir != env_dir or cls.env_id != env_id:
                cls.env_dir = env_dir
                cls.env_id = env_id
                print(f"Updating registration for Unity environment: {env_id} with file path: {env_dir}")
            else:
                print(f"Unity environment {env_id} already registered with the same file path; skipping registration.")
        else:
            cls.env_dir = env_dir
            cls.env_id = env_id
            print(f"Registering Unity environment: {env_id} with file path: {env_dir}")

    @staticmethod
    def make(env_id, **kwargs):
        return UnityEnv(env_id=env_id, is_vectorized=False, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        return UnityEnv(env_id=env_id, num_envs=num_envs, is_vectorized=True, **kwargs)

    def __init__(self, env_id, num_envs=1, is_vectorized=False, **kwargs):
        """
        Initialize a Unity environment.

        :param env_id: Path to the Unity environment binary.
        :param num_envs: Number of environments (for vectorized environments).
        :param worker_id: Unique identifier for multiple Unity instances.
        :param use_graphics: Whether to run with graphics.
        :param is_vectorized: Whether the environment is vectorized.
        :param seed: Random seed for the environment.
        :param time_scale: Time scale for the Unity environment.
        """
        super().__init__()
        self.seed = UnityEnv._instance_count
        UnityEnv._instance_count += num_envs

        use_graphics = kwargs.get("use_graphics", False)
        time_scale = kwargs.get("time_scale", 4)
        requested_worker_id = kwargs.get("worker_id")
        if UnityEnv.env_dir is None:
            raise ValueError("Unity environment not registered. Use UnityEnv.register() to register the environment.")
        else:
            if env_id != UnityEnv.env_id:
                print(f"Environment ID mismatch: {env_id} != {UnityEnv.env_id}")

        self.env_id = env_id

        self.num_envs = num_envs
        self.is_vectorized = is_vectorized
        self.no_graphics = not use_graphics
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(width=1280, height=720, time_scale=time_scale)

        if is_vectorized:
            first_worker_id = self.seed if requested_worker_id is None else int(requested_worker_id)
            # Create multiple environments without graphics for performance
            self.envs = [
                self.create_unity_env(
                    channel=self.channel,
                    no_graphics=True,
                    seed=self.seed + i,
                    worker_id=first_worker_id + i
                ) for i in range(num_envs)
            ]
        else:
            worker_id = self.seed + 100 if requested_worker_id is None else int(requested_worker_id)
            self.env = self.create_unity_env(
                channel=self.channel,
                no_graphics=self.no_graphics,
                seed=self.seed + 100,
                worker_id=worker_id
            )
            self.envs = [self.env]  # For consistency, make self.envs a list

        # A "group" = one (env instance, behavior) pair. Single-agent envs have
        # one group per env; a multi-behavior env like SoccerTwos has several (one
        # per team, e.g. SoccerTwos?team=0 / ?team=1). Agents are numbered globally
        # across all groups in `group_keys` order, and every map below is keyed by
        # the (env_idx, behavior_name) group key.
        self.specs = []            # per-group behavior spec (consistency-checked)
        self.behaviors_by_env = [] # env_idx -> [behavior_name, ...]
        self.group_keys = []       # ordered [(env_idx, behavior_name), ...]
        self.slot_maps = {}        # group -> {unity_agent_id: 0-based slot}
        self.offsets = {}          # group -> first global agent index
        self.counts = {}           # group -> agent count
        # agent_ids that requested a decision at the last poll, in Unity's order,
        # so set_actions rows line up with the agents Unity is asking about.
        self.decision_ids = {}     # group -> [unity_agent_id, ...]
        # Stable identity/context for every global agent slot. Group IDs are
        # supplied by ML-Agents and let collectors retain multi-agent match
        # structure instead of treating linked trajectories as unrelated.
        self.agent_context = []
        # Per cooperative group, latch newly issued IDs belonging to the current
        # reset. DungeonEscape can surface the three new decision IDs on
        # different ticks; only the first one marks a new scene boundary.
        self._reissued_ids = {}
        self.num_agents = 0
        self._initialize_env_info()
        self._define_observation_space()
        self._define_action_space()

    @staticmethod
    def create_unity_env(channel, no_graphics, seed, worker_id):
        base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT
        env = UnityEnvironment(
            file_name=UnityEnv.env_dir,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[channel],
            worker_id=worker_id,
        )
        return env

    def _initialize_env_info(self):
        total_agents = 0
        # Enumerate every (env, behavior) group. Unity numbers agent_ids globally
        # across behaviors, so sort each group's ids for a stable 0-based slot order
        # rather than assuming they are 0..n-1. All behaviors are driven and
        # recorded; multi-behavior envs (e.g. SoccerTwos' two teams) simply become
        # multiple groups sharing one Unity instance and one env.step().
        for env_idx, env in enumerate(self.envs):
            env.reset()
            bnames = sorted(env.behavior_specs.keys())
            self.behaviors_by_env.append(bnames)
            for bname in bnames:
                self.specs.append(env.behavior_specs[bname])
                decision_steps, _ = env.get_steps(bname)
                agent_ids = sorted(int(a) for a in decision_steps.agent_id)
                group_by_agent = {
                    int(a): int(g)
                    for a, g in zip(decision_steps.agent_id, decision_steps.group_id)
                }
                key = (env_idx, bname)
                self.group_keys.append(key)
                self.slot_maps[key] = {aid: slot for slot, aid in enumerate(agent_ids)}
                self.offsets[key] = total_agents
                self.counts[key] = len(agent_ids)
                self.decision_ids[key] = []
                team_id = int(bname.rsplit("?team=", 1)[1]) if "?team=" in bname else 0
                for agent_id in agent_ids:
                    self.agent_context.append(
                        {
                            "env_index": env_idx,
                            "behavior_name": bname,
                            "team_id": team_id,
                            "agent_id": agent_id,
                            "group_id": group_by_agent[agent_id],
                        }
                    )
                total_agents += len(agent_ids)

        self.num_agents = total_agents

    def _define_observation_space(self):
        # Check consistency of observation shapes across all specs
        reference_shapes = [obs_spec.shape for obs_spec in self.specs[0].observation_specs]
        for spec in self.specs[1:]:
            current_shapes = [obs_spec.shape for obs_spec in spec.observation_specs]
            if reference_shapes != current_shapes:
                raise ValueError("Observation shapes are inconsistent across specs.")

        # Define the observation space per agent
        observation_shapes = reference_shapes  # Use the consistent shapes from the first spec

        # Check if any observation shape is an image
        if any(len(shape) == 3 for shape in observation_shapes):
            raise ValueError("Image observations are not supported.")

        # Since the assumption of RayPerceptionSensor is removed,
        # default to an infinite observation range for all observations
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, *obs_spec.shape)
                        if isinstance(obs_spec.shape, tuple)
                        else (self.num_agents, obs_spec.shape),
                    dtype=np.float32
                )
                for obs_spec in self.specs[0].observation_specs
            ]
        )
        self.observation_shapes = observation_shapes

    def _define_action_space(self, start=0):
        # Ensure consistency of action specifications across all specs
        reference_action_spec = self.specs[0].action_spec
        for spec in self.specs[1:]:
            if spec.action_spec != reference_action_spec:
                raise ValueError("Action specifications are inconsistent across specs.")

        # Assume all environments have the same action space
        self.spec = self.specs[0]  # Use the first spec as the reference
        action_spec = self.spec.action_spec

        # Define the action space per agent
        if action_spec.continuous_size > 0 and action_spec.discrete_size == 0:
            # Continuous actions only
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
        elif action_spec.discrete_size > 0 and action_spec.continuous_size == 0:
            # Discrete actions only
            if action_spec.discrete_size == 1:
                # Single discrete action branch
                self.action_space = spaces.Discrete(
                    action_spec.discrete_branches[0] - start, start=start
                )
            else:
                # Multiple discrete action branches. discrete_branches is a plain
                # tuple in mlagents_envs 1.1.0, so coerce to an array; the branches
                # are the 0-based nvec directly (SoccerTwos = first env here).
                # gymnasium's MultiDiscrete rejects a scalar start, so leave it
                # 0-based (matches the onnx's 0-based indices and the dataset).
                self.action_space = spaces.MultiDiscrete(np.asarray(action_spec.discrete_branches))
        elif action_spec.continuous_size > 0 and action_spec.discrete_size > 0:
            # Mixed actions: Combine continuous and discrete spaces using Tuple
            continuous_space = spaces.Box(
                low=-1,
                high=1,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
            if action_spec.discrete_size == 1:
                discrete_space = spaces.Discrete(
                    action_spec.discrete_branches[0] - start, start=start
                )
            else:
                discrete_space = spaces.MultiDiscrete(np.asarray(action_spec.discrete_branches))
            self.action_space = spaces.Tuple((continuous_space, discrete_space))
        else:
            raise NotImplementedError("Action space configuration not supported.")

    def _create_action_tuple(self, actions, env_idx):
        action_tuple = ActionTuple()
        num_agents = len(actions)

        if num_agents == 0:
            # No agents to act upon
            return action_tuple

        if isinstance(self.action_space, spaces.Box):
            # Continuous actions only
            actions = np.asarray(actions, dtype=np.float32).reshape(num_agents, -1)
            action_tuple.add_continuous(actions)
        elif isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
            # Discrete actions only
            actions = np.asarray(actions, dtype=np.int32).reshape(num_agents, -1)
            action_tuple.add_discrete(actions)
        elif isinstance(self.action_space, spaces.Tuple):
            # Hybrid: `actions` is a [num_agents, continuous_size + num_branches]
            # row — continuous values first, then one 0-based index column per
            # discrete branch (the layout OnnxPolicy.act() emits for kind="hybrid").
            # Split by continuous_size and feed both halves of the ActionTuple.
            cont_size = self.spec.action_spec.continuous_size
            actions = np.asarray(actions, dtype=np.float32).reshape(num_agents, -1)
            continuous_action = actions[:, :cont_size].astype(np.float32)
            discrete_action = actions[:, cont_size:].astype(np.int32)
            action_tuple.add_continuous(continuous_action)
            action_tuple.add_discrete(discrete_action)
        else:
            raise NotImplementedError("Action type not supported.")

        return action_tuple

    def init_transitions(self, obs_len):

        num_agents = self.num_agents

        observations = tuple([[None] * num_agents for _ in range(obs_len)])
        final_observations = tuple([[None] * num_agents for _ in range(obs_len)])

        # Initialize other transition variables
        rewards = [None] * num_agents
        terminated = [None] * num_agents
        truncated = [None] * num_agents

        return observations, rewards, terminated, truncated, final_observations

    def _resolve_step_slots(self, key, decision_steps, terminal_steps):
        """Resolve possibly reissued Unity agent IDs to stable wrapper slots.

        Some cooperative examples (notably DungeonEscape) register the same
        GameObjects with ``SimpleMultiAgentGroup`` again after every reset. That
        gives them new ML-Agents agent IDs even though the number of agents and
        their group membership are unchanged. Keep dataset slots stable by
        assigning each unseen ID to a free slot belonging to the same group.
        """
        slot_map = self.slot_maps[key]
        offset = self.offsets[key]
        step_slot_map = {}
        unseen_decision_groups = {}
        for agent_id, group_id in zip(decision_steps.agent_id, decision_steps.group_id):
            if int(agent_id) not in slot_map:
                group_id = int(group_id)
                unseen_decision_groups.setdefault(group_id, set()).add(int(agent_id))

        terminal_used = set()
        for raw_agent_id, raw_group_id in zip(
            terminal_steps.agent_id, terminal_steps.group_id
        ):
            agent_id = int(raw_agent_id)
            group_id = int(raw_group_id)
            slot = slot_map.get(agent_id)
            if slot is None or slot in terminal_used:
                candidates = [
                    local_slot
                    for local_slot in range(self.counts[key])
                    if self.agent_context[offset + local_slot]["group_id"] == group_id
                    and local_slot not in terminal_used
                ]
                if not candidates:
                    raise RuntimeError(
                        f"No free stable slot for agent_id={agent_id}, group_id={group_id}, "
                        f"behavior={key}; group membership may have changed."
                    )
                slot = candidates[0]
                slot_map[agent_id] = slot
            step_slot_map[agent_id] = slot
            terminal_used.add(slot)

        # A reset can expose an old terminal ID and a newly issued decision ID
        # for the same physical agent on one poll. Decision rows may therefore
        # intentionally reuse a terminal slot, but must remain unique among
        # themselves.
        decision_used = set()
        for raw_agent_id, raw_group_id in zip(
            decision_steps.agent_id, decision_steps.group_id
        ):
            agent_id = int(raw_agent_id)
            group_id = int(raw_group_id)
            slot = slot_map.get(agent_id)
            if slot is None or slot in decision_used:
                candidates = [
                    local_slot
                    for local_slot in range(self.counts[key])
                    if self.agent_context[offset + local_slot]["group_id"] == group_id
                    and local_slot not in decision_used
                ]
                if not candidates:
                    raise RuntimeError(
                        f"No free stable decision slot for agent_id={agent_id}, "
                        f"group_id={group_id}, behavior={key}."
                    )
                # Prefer recycling a slot whose old ID terminates on this poll.
                terminal_candidates = [slot for slot in candidates if slot in terminal_used]
                slot = terminal_candidates[0] if terminal_candidates else candidates[0]
                slot_map[agent_id] = slot
            step_slot_map[agent_id] = slot
            decision_used.add(slot)
        group_sizes = {}
        for local_slot in range(self.counts[key]):
            group_id = self.agent_context[offset + local_slot]["group_id"]
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
        restarted_group_ids = set()
        for group_id, new_ids in unseen_decision_groups.items():
            pending_key = (key, group_id)
            pending = self._reissued_ids.setdefault(pending_key, set())
            if not pending:
                restarted_group_ids.add(group_id)
            pending.update(new_ids)
            if len(pending) >= group_sizes.get(group_id, 0):
                pending.clear()
        return step_slot_map, restarted_group_ids


    def reset(self, **kwargs):
        """
        Reset the Unity environment(s) and retrieve initial observations.
        :return: Initial aggregated observations and info dictionary.
        """
        obs_len = len(self.observation_shapes)
        observations = tuple([[None] * self.num_agents for _ in range(obs_len)])
        for env_idx, env in enumerate(self.envs):
            env.reset()
            for bname in self.behaviors_by_env[env_idx]:
                key = (env_idx, bname)
                slot_map = self.slot_maps[key]
                offset = self.offsets[key]
                decision_steps, _ = env.get_steps(bname)

                self.decision_ids[key] = [int(a) for a in decision_steps.agent_id]
                if len(decision_steps.agent_id) == 0:
                    # No agents to act upon
                    continue
                obs = decision_steps.obs
                for idx, agent_id in enumerate(decision_steps.agent_id):
                    global_idx = offset + slot_map[int(agent_id)]
                    # Aggregate all observation components
                    for i in range(obs_len):
                        observations[i][global_idx] = obs[i][idx]

        return observations, {}

    def step(self, actions):
        """
        Perform a step in the Unity environment(s).
        :param actions: Actions to take for all agents.
        :return: Tuple containing observations, rewards, terminated flags, truncated flags, and info.
        """
        # Set actions for every behavior in each env, then advance that env once.
        # env.step() drives the whole Unity instance, so it runs per env, after all
        # of that instance's behaviors (e.g. both SoccerTwos teams) have their
        # actions set.
        for env_idx, env in enumerate(self.envs):
            for bname in self.behaviors_by_env[env_idx]:
                key = (env_idx, bname)
                slot_map = self.slot_maps[key]
                # This group's slice of the global action array, indexed by slot.
                env_actions = actions[self.offsets[key]:self.offsets[key] + self.counts[key]]

                # Build action rows in Unity's decision order (decision_ids), each
                # pulled from its agent's slot, so set_actions lines rows up with the
                # agents Unity is asking about regardless of the id layout.
                dec_ids = self.decision_ids[key]
                if len(dec_ids) > 0:
                    dec_actions = np.stack(
                        [env_actions[slot_map[aid]] for aid in dec_ids], axis=0
                    )
                    action_tuple = self._create_action_tuple(dec_actions, env_idx)
                    env.set_actions(bname, action_tuple)
                self.decision_ids[key] = []
            env.step()

        obs_len = len(self.observation_shapes)
        observations, rewards, terminated, truncated, final_observations = self.init_transitions(obs_len)
        restarted_groups = []
        # Collect results from every (env, behavior) group.
        for env_idx, env in enumerate(self.envs):
            for bname in self.behaviors_by_env[env_idx]:
                key = (env_idx, bname)
                slot_map = self.slot_maps[key]
                offset = self.offsets[key]
                decision_steps, terminal_steps = env.get_steps(bname)
                step_slot_map, restarted_group_ids = self._resolve_step_slots(
                    key, decision_steps, terminal_steps
                )
                restarted_groups.extend(
                    (env_idx, bname, group_id) for group_id in restarted_group_ids
                )
                # Record who requested a decision (Unity order) for the next step's
                # set_actions, before the empty-step early-out below.
                self.decision_ids[key] = [int(a) for a in decision_steps.agent_id]
                if len(decision_steps.agent_id) == 0 and len(terminal_steps.agent_id) == 0:
                    # Nothing this step: between-decision (action-repeat) gap with no
                    # terminations. Only skip when BOTH are empty.
                    continue
                # Do NOT skip when only decision_steps is empty: an agent can land in
                # terminal_steps on a step where no agent has a decision (it terminates
                # during an action-repeat gap). Skipping here silently drops that
                # termination -> missed episode boundary -> merged/over-long episodes.
                # With an empty decision set, common/decision_only below collapse to
                # empty and terminal_only captures every terminal agent, so fall through.

                # Merge by stable slot, not raw agent ID. On a DungeonEscape
                # reset the old terminal ID and new decision ID can differ while
                # referring to the same physical slot.
                decision_by_slot = {
                    step_slot_map[int(agent_id)]: idx
                    for idx, agent_id in enumerate(decision_steps.agent_id)
                }
                terminal_by_slot = {
                    step_slot_map[int(agent_id)]: idx
                    for idx, agent_id in enumerate(terminal_steps.agent_id)
                }
                dec_obs = decision_steps.obs
                term_obs = terminal_steps.obs
                for slot in set(decision_by_slot) | set(terminal_by_slot):
                    global_idx = offset + slot
                    dec_local_idx = decision_by_slot.get(slot)
                    term_local_idx = terminal_by_slot.get(slot)
                    if dec_local_idx is not None:
                        for i in range(obs_len):
                            observations[i][global_idx] = dec_obs[i][dec_local_idx]
                    if term_local_idx is None:
                        rewards[global_idx] = float(
                            decision_steps.reward[dec_local_idx]
                            + decision_steps.group_reward[dec_local_idx]
                        )
                        terminated[global_idx] = False
                        truncated[global_idx] = False
                        continue

                    if dec_local_idx is not None:
                        for i in range(obs_len):
                            final_observations[i][global_idx] = term_obs[i][term_local_idx]
                    else:
                        # No next-episode decision observation yet; retain the
                        # existing terminal-only contract used by the collector.
                        for i in range(obs_len):
                            observations[i][global_idx] = term_obs[i][term_local_idx]
                    rewards[global_idx] = float(
                        terminal_steps.reward[term_local_idx]
                        + terminal_steps.group_reward[term_local_idx]
                    )
                    if terminal_steps.interrupted[term_local_idx]:
                        truncated[global_idx] = True  # Adjust if necessary
                        terminated[global_idx] = False
                    else:
                        terminated[global_idx] = True
                        truncated[global_idx] = False

        info = {}
        info['final_observation'] = final_observations
        info['restarted_groups'] = restarted_groups

        return observations, rewards, terminated, truncated, info

    def close(self):
        """
        Close the Unity environment(s).
        """
        for env in self.envs:
            env.close()

        self.envs = []

    def __exit__(self, exc_type, exc_value, traceback):
        if self.envs:
            self.close()
