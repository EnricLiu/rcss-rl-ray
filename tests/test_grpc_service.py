"""Unit tests for the upgraded GameServicer with per-unum dispatch."""

from __future__ import annotations

import threading
import pytest

from rcss_rl.proto import service_pb2 as pb2
from rcss_rl.env.grpc_service import GameServicer


def _make_state(unum: int, cycle: int = 1) -> pb2.State:
    """Create a minimal State message for testing."""
    wm = pb2.WorldModel(
        cycle=cycle,
        self=pb2.Self(id=unum, uniform_number=unum),
        ball=pb2.Ball(position=pb2.Vector2D(x=0.0, y=0.0)),
    )
    return pb2.State(world_model=wm)


class TestGameServicerRegistration:
    def test_register_player(self) -> None:
        s = GameServicer()
        s.register_player(2)
        s.register_player(5)
        assert s.registered_unums == frozenset({2, 5})

    def test_registered_unums_empty(self) -> None:
        s = GameServicer()
        assert s.registered_unums == frozenset()


class TestGameServicerPerUnum:
    def test_wait_for_state_unregistered_raises(self) -> None:
        s = GameServicer()
        with pytest.raises(RuntimeError, match="not registered"):
            s.wait_for_state(99)

    def test_set_actions_unregistered_raises(self) -> None:
        s = GameServicer()
        with pytest.raises(RuntimeError, match="not registered"):
            s.set_actions(99, pb2.PlayerActions())

    def test_state_action_round_trip(self) -> None:
        """Simulate a full GetPlayerActions round-trip for one player."""
        s = GameServicer()
        s.register_player(3)

        result_holder: list[pb2.PlayerActions] = []
        state = _make_state(unum=3, cycle=10)

        def sidecar_thread() -> None:
            # Simulate the sidecar calling GetPlayerActions.
            result = s.GetPlayerActions(state, context=None)
            result_holder.append(result)

        t = threading.Thread(target=sidecar_thread)
        t.start()

        # Wait for the state to arrive in the servicer.
        assert s.wait_for_state(3, timeout=2.0)

        # Verify stored world model.
        wm = s.get_world_model(3)
        assert wm is not None
        assert wm.cycle == 10
        assert wm.self.id == 3

        # Push actions back.
        actions = pb2.PlayerActions(
            actions=[pb2.PlayerAction(dash=pb2.Dash(power=50.0))]
        )
        s.set_actions(3, actions)
        t.join(timeout=5.0)
        assert not t.is_alive()

        # Verify sidecar received the actions.
        assert len(result_holder) == 1
        assert len(result_holder[0].actions) == 1
        assert result_holder[0].actions[0].dash.power == 50.0

    def test_multi_player_concurrent(self) -> None:
        """Two players exchange state/actions concurrently."""
        s = GameServicer()
        s.register_player(2)
        s.register_player(7)

        results: dict[int, pb2.PlayerActions] = {}

        def sidecar(unum: int) -> None:
            st = _make_state(unum=unum, cycle=5)
            result = s.GetPlayerActions(st, context=None)
            results[unum] = result

        t2 = threading.Thread(target=sidecar, args=(2,))
        t7 = threading.Thread(target=sidecar, args=(7,))
        t2.start()
        t7.start()

        # Wait for both states.
        assert s.wait_for_state(2, timeout=2.0)
        assert s.wait_for_state(7, timeout=2.0)

        # Send different actions to each player.
        s.set_actions(2, pb2.PlayerActions(
            actions=[pb2.PlayerAction(dash=pb2.Dash(power=20.0))]
        ))
        s.set_actions(7, pb2.PlayerActions(
            actions=[pb2.PlayerAction(kick=pb2.Kick(power=80.0))]
        ))

        t2.join(timeout=5.0)
        t7.join(timeout=5.0)
        assert not t2.is_alive()
        assert not t7.is_alive()

        assert results[2].actions[0].dash.power == 20.0
        assert results[7].actions[0].kick.power == 80.0

    def test_reset_clears_buffers(self) -> None:
        s = GameServicer()
        s.register_player(5)

        # Simulate a state arriving.
        def sidecar() -> None:
            s.GetPlayerActions(_make_state(5), context=None)

        t = threading.Thread(target=sidecar)
        t.start()
        assert s.wait_for_state(5, timeout=2.0)

        # Push actions to unblock sidecar.
        s.set_actions(5, pb2.PlayerActions())
        t.join(timeout=5.0)

        # Verify state is stored.
        assert s.get_state(5) is not None

        # Reset should clear states.
        s.reset()
        assert s.get_state(5) is None
        # But player is still registered.
        assert 5 in s.registered_unums

    def test_get_state_returns_none_for_unknown(self) -> None:
        s = GameServicer()
        assert s.get_state(99) is None

    def test_get_world_model_returns_none_for_unknown(self) -> None:
        s = GameServicer()
        assert s.get_world_model(99) is None
