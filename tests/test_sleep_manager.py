from datetime import timedelta

import pytest

from econsimulacra.envs import SleepManager
from econsimulacra.logs import SleepEndLog, SleepStartLog


class TestSleepManager:
    def test_init(self):
        config = {}
        sleep_manager = SleepManager(config)
        assert sleep_manager.config == config
        assert sleep_manager.registered_classes == []
        assert sleep_manager.agent_id2is_sleeping == {}
        assert sleep_manager.agent_id2since == {}
        assert sleep_manager.agent_id2until == {}

    def test_resolve_sleep_duration(self):
        config = {}
        sleep_manager = SleepManager(config)
        assert sleep_manager._resolve_sleep_duration("1h") == timedelta(hours=1)
        assert sleep_manager._resolve_sleep_duration("1.5h") == timedelta(
            hours=1, minutes=30
        )
        assert sleep_manager._resolve_sleep_duration("30m") == timedelta(minutes=30)
        with pytest.raises(ValueError):
            sleep_manager._resolve_sleep_duration("1d")
        with pytest.raises(ValueError):
            sleep_manager._resolve_sleep_duration("60")

    def test_update_sleep_status(self):
        config = {}
        sleep_manager = SleepManager(config)
        agent_id = 1
        current_time = "2024-01-01 00:00:00"
        current_time_step = 0
        log = sleep_manager.update_sleep_status(
            agent_id=agent_id,
            current_time=current_time,
            current_time_step=current_time_step,
        )
        assert log is None
        assert not sleep_manager.agent_id2is_sleeping[agent_id]
        assert sleep_manager.agent_id2since[agent_id] is None
        assert sleep_manager.agent_id2until[agent_id] is None
        current_time = "2024-01-01 01:00:00"
        current_time_step = 1
        log = sleep_manager.update_sleep_status(
            agent_id=agent_id,
            current_time=current_time,
            current_time_step=current_time_step,
            sleep_duration="2h",
        )
        assert isinstance(log, SleepStartLog)
        assert sleep_manager.agent_id2is_sleeping[agent_id]
        assert sleep_manager.agent_id2since[agent_id] == "2024-01-01 01:00:00"
        assert sleep_manager.agent_id2until[agent_id] == "2024-01-01 03:00:00"
        current_time = "2024-01-01 02:00:00"
        current_time_step = 2
        log = sleep_manager.update_sleep_status(
            agent_id=agent_id,
            current_time=current_time,
            current_time_step=current_time_step,
        )
        assert log is None
        assert sleep_manager.agent_id2is_sleeping[agent_id]
        assert sleep_manager.agent_id2since[agent_id] == "2024-01-01 01:00:00"
        assert sleep_manager.agent_id2until[agent_id] == "2024-01-01 03:00:00"
        with pytest.raises(ValueError):
            sleep_manager.update_sleep_status(
                agent_id=agent_id,
                current_time=current_time,
                current_time_step=current_time_step,
                sleep_duration="1h",
            )
        current_time = "2024-01-01 03:00:00"
        current_time_step = 3
        log = sleep_manager.update_sleep_status(
            agent_id=agent_id,
            current_time=current_time,
            current_time_step=current_time_step,
        )
        assert isinstance(log, SleepEndLog)
        assert not sleep_manager.agent_id2is_sleeping[agent_id]
        assert sleep_manager.agent_id2since[agent_id] is None
        assert sleep_manager.agent_id2until[agent_id] is None
