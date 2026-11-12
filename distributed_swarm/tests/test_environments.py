import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from distributed_swarm.envs.swarm_2d import Swarm2D


def test_2d_gym_contract_and_determinism():
    env = Swarm2D(max_episode_steps=5)
    check_env(env, skip_render_check=True)
    def rollout():
        obs, _ = env.reset(seed=22)
        history = [obs]
        for _ in range(5):
            obs, r, terminated, truncated, info = env.step(np.full(5,4))
            history.append(obs)
        assert truncated
        return np.stack(history), info["network"]
    a, sa = rollout(); b, sb = rollout()
    np.testing.assert_array_equal(a,b)
    assert sa == sb
    env.close()


@pytest.mark.parametrize("num_agents,map_size", [(10,30), (20,45)])
def test_large_2d_worlds(num_agents, map_size):
    env = Swarm2D(num_agents=num_agents, map_size=map_size,
                  communication_radius=map_size * .4, max_episode_steps=2)
    obs, _ = env.reset(seed=17)
    assert obs.shape == (num_agents, 10 + 9 * (num_agents - 1))
    assert env.action_space.shape == (num_agents,)
    obs, reward, _, _, info = env.step(np.full(num_agents, 4))
    assert np.isfinite(obs).all() and np.isfinite(reward)
    assert info["agent_rewards"].shape == (num_agents,)
    env.close()


def test_neighbor_observation_has_no_ground_truth_access():
    env = Swarm2D(network=dict(packet_loss=0,max_delay=0))
    env.reset(seed=3)
    before = env._observations()[0].copy()
    env.positions[1:] += 100
    env.velocities[1:] += 100
    after = env._observations()[0]
    np.testing.assert_array_equal(before,after)
    # Other agents' own sensors change, as expected.
    assert not np.array_equal(env._observations()[1], before)


def test_crash_stays_stationary_and_stops_sending():
    env = Swarm2D(network=dict(packet_loss=0,max_delay=0,communication_radius=100,
        crashes=[dict(robot_id=0,time=2)]))
    env.reset(seed=1)
    for _ in range(2):
        env.step(np.full(5,3))
    position = env.positions[0].copy()
    seq = env.agents[0].clock.sequence_number
    env.step(np.full(5,3))
    np.testing.assert_array_equal(env.positions[0],position)
    assert env.agents[0].clock.sequence_number == seq
    assert env.crashed[0]


def test_partition_does_not_stop_local_actions_and_reintegrates():
    env = Swarm2D(network=dict(packet_loss=0,max_delay=0,communication_radius=100,
        disconnections=[dict(start=2,end=12,robots=[0])]))
    env.reset(seed=1)
    for _ in range(15):
        env.step(np.full(5,4))
    states = [e["status"] for e in env.agents[1].detector.events if e["robot"] == 0]
    assert states == ["SUSPECTED","UNAVAILABLE","RECOVERED"]
    assert not env.crashed.any()


def test_collision_with_crashed_robot_is_blocked():
    env = Swarm2D(num_agents=2,obstacles=[],network=dict(crashes=[dict(robot_id=0,time=0)]))
    env.reset(seed=0)
    env.positions[:] = [[2,2],[2.8,2]]
    env.velocities[:] = 0
    _,_,_,_,info = env.step(np.array([3,2]))
    np.testing.assert_allclose(env.positions,[[2,2],[2.8,2]])
    assert info["collisions"] == 2


def test_shared_ant_world_and_configurable_count():
    pytest.importorskip("mujoco")
    from distributed_swarm.envs.swarm_ant_3d import SwarmAnt3D
    for n in (2,5,10):
        env = SwarmAnt3D(num_agents=n,max_episode_steps=3)
        assert env.model.nq == 15*n
        assert env.model.nv == 14*n
        assert env.model.nu == 8*n
        check_env(env,skip_render_check=True)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (n,33+9*(n-1))
        for _ in range(3):
            obs,r,term,trunc,info = env.step(np.zeros((n,8),np.float32))
        assert np.isfinite(obs).all()
        assert np.isfinite(r)
        env.close()


def test_ant_crash_is_distinct_from_fall():
    pytest.importorskip("mujoco")
    from distributed_swarm.envs.swarm_ant_3d import SwarmAnt3D
    env = SwarmAnt3D(num_agents=2,network=dict(crashes=[dict(robot_id=0,time=0)]))
    env.reset(seed=1)
    qpos=env.data.qpos[:15].copy()
    _,_,_,_,info=env.step(np.ones((2,8),np.float32))
    np.testing.assert_array_equal(env.data.qpos[:15],qpos)
    assert info["crashed"][0] and not info["fallen"][0]
    env.close()


def test_ant_communication_velocity_uses_control_tick_units():
    from distributed_swarm.envs.swarm_ant_3d import SwarmAnt3D
    env = SwarmAnt3D(num_agents=1,frame_skip=5)
    env.reset(seed=1)
    env.data.qvel[:2] = [2.,-1.]
    env._read_state()
    np.testing.assert_allclose(env.velocities[0],[.1,-.05])
    env.close()


def test_ant_discrete_direction_action_space_and_mapping():
    from distributed_swarm.envs.swarm_ant_3d import SwarmAnt3D
    env = SwarmAnt3D(num_agents=5, control_mode="discrete_direction",
                     max_episode_steps=2, fall_as_failure=False)
    assert env.action_space.shape == (5,)
    np.testing.assert_array_equal(env.action_space.nvec, np.full(5, 5))
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(info["direction_commands"],
        [[0,1], [0,-1], [-1,0], [1,0], [0,0]])
    assert info["motor_torques"].shape == (5, 8)
    assert np.isfinite(reward)
    with pytest.raises(ValueError):
        env.step(np.array([0, 1, 2, 3, 5]))
    env.close()


def test_ant_robots_can_physically_collide():
    from distributed_swarm.envs.swarm_ant_3d import SwarmAnt3D
    env = SwarmAnt3D(num_agents=2)
    env.reset(seed=1)
    env.data.qpos[15:17] = env.data.qpos[:2] + [.2,0.]
    import mujoco
    mujoco.mj_forward(env.model,env.data)
    pairs = [(env.geom_owners[c.geom1],env.geom_owners[c.geom2]) for c in env.data.contact]
    assert any(a >= 0 and b >= 0 and a != b for a,b in pairs)
    env.close()
