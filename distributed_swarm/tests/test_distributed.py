import numpy as np
import pytest
from distributed_swarm.distributed.communication import RobotMessage, NetworkConfig, CommunicationSimulator
from distributed_swarm.distributed.logical_clock import LamportClock
from distributed_swarm.distributed.neighbor_cache import NeighborCache
from distributed_swarm.distributed.failure_detector import FailureDetector
from distributed_swarm.distributed.consistency import ConsistencyConfig, estimate
from distributed_swarm.agents.robot_agent import RobotAgent


def packet(seq=1, sent=0., receiver=1, kind="state"):
    return RobotMessage(0, receiver, seq, seq+3, sent, np.array([1.,2.]), np.array([.5,0.]), kind)


def test_lamport_send_receive():
    clock = LamportClock()
    assert clock.send() == (1,1)
    clock.receive(10)
    assert clock.value == 11
    assert clock.send() == (12,2)


@pytest.mark.parametrize("filtering", [True,False])
def test_stale_sequence_and_physical_timestamp(filtering):
    cache = NeighborCache(filtering)
    cache.update(packet(5, 1.))
    cache.update(packet(3, .5))
    cache.update(packet(4, .8))
    assert cache.stale_received == 2
    assert cache.stale_rejected == (2 if filtering else 0)
    assert cache.entries[0].sequence_number == (5 if filtering else 4)
    assert cache.entries[0].last_update_time == (1. if filtering else .8)


def test_queue_seed_snapshot_and_reordering():
    def simulate():
        network = CommunicationSimulator(NetworkConfig(packet_loss=0, max_delay=5), seed=3)
        for k in range(10):
            msg = packet(k)
            network.send(msg, 1.)
            msg.position[:] = 100
        assert not network.deliver(-1.)
        result = network.deliver(10.)
        assert all(np.array_equal(m.position, [1.,2.]) for m in result)
        return [m.sequence_number for m in result]
    assert simulate() == simulate()
    assert simulate() != sorted(simulate())
    fifo = CommunicationSimulator(NetworkConfig(packet_loss=0, max_delay=5, reorder=False), 3)
    for k in range(10):
        fifo.send(packet(k), 1.)
    assert [m.sequence_number for m in fifo.deliver(10)] == list(range(10))


def test_partition_recovery_crash_range_and_loss():
    cfg = NetworkConfig(packet_loss=0, max_delay=0, partitions=[dict(start=2,end=4,groups=[[0],[1]])],
                        crashes=[dict(robot_id=0,time=6)])
    network = CommunicationSimulator(cfg, 1)
    for time, expected in [(0,1),(2,0),(4,1),(6,0)]:
        network.send(packet(sent=time), 1)
        assert len(network.deliver(time)) == expected
    network.send(packet(sent=7,receiver=2), 100)
    assert not network.deliver(100)
    lossy = CommunicationSimulator(NetworkConfig(packet_loss=1), 1)
    lossy.send(packet(), 0)
    assert not lossy.deliver(100)


def test_inflight_partition_drop():
    net = CommunicationSimulator(NetworkConfig(packet_loss=0,min_delay=3,max_delay=3,
        disconnections=[dict(start=2,end=4,robots=[0])]),1)
    net.send(packet(),1)
    assert not net.deliver(3)
    assert net.stats["dropped"] == 1


def test_detector_suspicion_unavailable_recovery_and_delayed_heartbeat():
    detector = FailureDetector(2,4)
    detector.heartbeat(0,0,10,0)
    detector.tick(3)
    assert detector.status(0) == "SUSPECTED"
    detector.tick(5)
    assert detector.status(0) == "UNAVAILABLE"
    detector.heartbeat(0,0,9,5)
    assert detector.status(0) == "UNAVAILABLE"
    detector.heartbeat(0,5,11,5)
    assert detector.status(0) == "RECOVERED"
    detector.tick(5)
    assert detector.status(0) == "ACTIVE"
    assert [e["status"] for e in detector.events] == ["SUSPECTED","UNAVAILABLE","RECOVERED"]


def test_confidence_prediction_use_sender_time():
    cache = NeighborCache(); cache.update(packet(1,2.))
    position, age, confidence = estimate(cache.entries[0], 6., "ACTIVE",
        ConsistencyConfig(predict_position=True,lambda_decay=.5))
    np.testing.assert_allclose(position, [3.,2.])
    assert age == 4
    assert confidence == pytest.approx(np.exp(-2))
    assert estimate(cache.entries[0],6,"UNAVAILABLE",ConsistencyConfig())[2] == 0


def test_clocks_do_not_change_navigation_state():
    agents = [RobotAgent(1,ConsistencyConfig(logical_clocks=enabled)) for enabled in (False,True)]
    for agent in agents:
        agent.receive(packet(4),2)
        agent.receive(packet(3),3)
    np.testing.assert_array_equal(agents[0].cache.entries[0].position,agents[1].cache.entries[0].position)
    assert agents[0].cache.stale_rejected == agents[1].cache.stale_rejected == 1
    assert agents[0].clock.value == 0 < agents[1].clock.value


def test_newer_but_physically_stale_heartbeat_cannot_recover():
    detector = FailureDetector(2,4)
    detector.heartbeat(0,0,1,0)
    detector.tick(6)
    assert detector.status(0) == "UNAVAILABLE"
    detector.heartbeat(0,1,2,6)
    detector.tick(6)
    assert detector.status(0) == "UNAVAILABLE"
    assert not any(e["status"] == "RECOVERED" for e in detector.events)
