import json
import os
import pathlib
from pathlib import Path
from typing import Any, Optional

from manim import Scene


class Animator(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log_txt_path: Path = pathlib.Path(os.environ["LOG_TXT_PATH"])
        config_path: Path = pathlib.Path(os.environ["CONFIG_PATH"])
        config: dict[str, Any] = json.load(open(config_path, "r"))
        self._get_log_configs(config)
        with open(log_txt_path, "r") as f:
            self.log_dics: list[dict[str, Any]] = [
                json.loads(line) for line in f.readlines()
            ]

    def _match_names(self, agent_name: str, name_keys: list[str]) -> str:
        for name_key in name_keys:
            if name_key in agent_name:
                return name_key
        return ""

    def construct(self) -> None:
        pass

    def _get_agent_id2name_dic(self) -> dict[int, str]:
        agent_id2name_dic: dict[int, str] = {}
        for log_dic in self.log_dics:
            if log_dic.get("type") == "agent_generation":
                agent_id: int = log_dic["agent_id"]
                agent_name: str = log_dic["agent_name"]
                agent_id2name_dic[agent_id] = agent_name
        return agent_id2name_dic

    def _get_log_configs(self, config: dict[str, Any]) -> None:
        """Get additional configurations from the config file.

        Args:
            config (dict[str, Any]): The full simulation config dict. Expected keys include
                ``simulation.numSteps``, ``environment.agents``, and per-agent-type dicts
                with optional ``svgPath`` and ``iconScale`` entries.

        Example config structure::

            {
                "simulation": {"numSteps": int},
                "environment": {
                    "gridSpace": [int, ...],
                    "cashName": str,
                    "agents": ["Household", "Retailer", ...]
                },
                "Household": {"svgPath": str, "isHousehold": bool, ...},
                "animation": {"cellSize": int, "drawMarginCells": int}
            }
        """
        assert "simulation" in config
        assert "numSteps" in config["simulation"]
        self.num_steps: int = config["simulation"]["numSteps"]
        assert "environment" in config
        agent_types: list[str] = config["environment"]["agents"]
        self.name2svg_path_dic: dict[str, Optional[str]] = {}
        self.name2icon_scale_dic: dict[str, float] = {}
        for agent_type in agent_types:
            agent_config: dict[str, Any] = config.get(agent_type, {})
            svg_path: Optional[str] = agent_config.get("svgPath", None)
            if svg_path is not None:
                self.name2svg_path_dic[agent_type] = svg_path
            icon_scale: float = agent_config.get("iconScale", 1.0)
            self.name2icon_scale_dic[agent_type] = icon_scale
        self.animation_config: dict[str, Any] = config.get("animation", {})
