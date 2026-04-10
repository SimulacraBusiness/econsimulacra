from datetime import datetime, timedelta

from econsimulacra.envs import TimeTranslator


class TestTimeTranslator:
    config: dict[str, str | int] = {
        "numSteps": 10,
        "startDatetime": "2025-01-01 00:00:00",
        "endDatetime": "2025-01-01 00:10:00",
    }

    def test_init(self) -> None:
        time_translator = TimeTranslator(config=self.config)
        assert time_translator.num_steps == self.config["numSteps"]
        assert time_translator.start_datetime == datetime.strptime(
            self.config["startDatetime"], "%Y-%m-%d %H:%M:%S"
        )
        assert time_translator.end_datetime == datetime.strptime(
            self.config["endDatetime"], "%Y-%m-%d %H:%M:%S"
        )
        assert time_translator.time_delta == timedelta(minutes=1)

    def test_step_to_datetime(self) -> None:
        time_translator = TimeTranslator(config=self.config)
        for step in range(self.config["numSteps"]):
            expected_datetime = (
                time_translator.start_datetime + timedelta(minutes=step)
            ).strftime("%Y-%m-%d %H:%M:%S")
            assert time_translator.step_to_datetime(step) == expected_datetime

    def test_get_timedelta(self) -> None:
        time_translator = TimeTranslator(config=self.config)
        assert time_translator.get_timedelta() == "0:01:00"
