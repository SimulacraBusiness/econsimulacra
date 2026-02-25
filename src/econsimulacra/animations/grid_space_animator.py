from .base import Animator
from collections import defaultdict
from manim import AnimationGroup
from manim import Dot
from manim import FadeIn
from manim import LaggedStart
from manim import Mobject
from manim import NumberPlane
from manim import SVGMobject
from manim import Text
from manim import Transform
from manim import VGroup
from manim.constants import DL, ORIGIN, UL, UP, UR
import numpy as np
from numpy import ndarray
from typing import Any
from typing import Optional
from typing import Tuple


Pos = Tuple[int, int]


class GridMapper:
    def __init__(
        self,
        number_plane: NumberPlane,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
    ) -> None:
        self.number_plane: NumberPlane = number_plane
        self.xmin: int = xmin
        self.xmax: int = xmax
        self.ymin: int = ymin
        self.ymax: int = ymax
        bottom_left: tuple[int, ...] = self.number_plane.get_corner(DL)
        self.bottom_left: ndarray = np.array(bottom_left)
        top_right: tuple[int, ...] = self.number_plane.get_corner(UR)
        np_width: float = top_right[0] - bottom_left[0]
        np_height: float = top_right[1] - bottom_left[1]
        num_cells_x: int = self.xmax - self.xmin + 1
        num_cells_y: int = self.ymax - self.ymin + 1
        self.cell_width: float = np_width / num_cells_x
        self.cell_height: float = np_height / num_cells_y

    def map_pos_to_manim_coord(self, pos: Pos) -> ndarray:
        return self.bottom_left + np.array(
            [
                (pos[0] - self.xmin + 0.5) * self.cell_width,
                (pos[1] - self.ymin + 0.5) * self.cell_height,
                0.0,
            ]
        )


class GridSpaceAnimator(Animator):
    def _get_log_configs(self, config: dict[str, Any]):
        super()._get_log_configs(config)
        assert "gridSpace" in config["environment"]
        if len(config["environment"]["gridSpace"]) != 2:
            raise ValueError("Only 2D grid space is supported in GridSpaceAnimator.")
        self.space_width: int = config["environment"]["gridSpace"][0]
        self.space_height: int = config["environment"]["gridSpace"][1]
        self.cell_size: int = self.animation_config.get("cellSize", 1)
        self.draw_margin_cells: int = self.animation_config.get("drawMarginCells", 2)

    def _get_agent_id2init_pos_dic(self) -> dict[int, Pos]:
        agent_id2init_pos_dic: dict[int, Pos] = {}
        for log_dic in self.log_dics:
            if log_dic.get("type") == "space_assign":
                agent_id: int = log_dic["agent_id"]
                pos: Pos = tuple(log_dic["pos"])  # type: ignore
                agent_id2init_pos_dic[agent_id] = pos
        return agent_id2init_pos_dic

    def _get_time2move_dic(
        self,
    ) -> tuple[list[str | int], dict[int, list[tuple[int | str, Pos, Pos]]]]:
        time2move_dic: dict[int | str, list[tuple[int | str, Pos, Pos]]] = defaultdict(
            list
        )
        times: list[int | str] = []
        for log_dic in self.log_dics:
            if log_dic.get("type") == "move":
                time: int | str = log_dic["time"]
                if time not in times:
                    times.append(time)
                agent_id: int = log_dic["agent_id"]
                old_pos: Pos = tuple(log_dic["old_pos"])  # type: ignore
                new_pos: Pos = tuple(log_dic["new_pos"])  # type: ignore
                time2move_dic[time].append((agent_id, old_pos, new_pos))
        return times, time2move_dic

    def construct(self) -> None:
        self.agent_id2name_dic: dict[int, str] = self._get_agent_id2name_dic()
        self.agent_id2init_pos_dic: dict[int, Pos] = self._get_agent_id2init_pos_dic()
        self.agent_ids: list[int] = sorted(list(self.agent_id2init_pos_dic.keys()))
        self.times, self.time2move_dic = self._get_time2move_dic()
        self.times: list[int | str] = sorted(self.times)
        xmin: int
        xmax: int
        ymin: int
        ymax: int
        xmin, xmax = (
            -self.draw_margin_cells,
            self.space_width - 1 + self.draw_margin_cells,
        )
        ymin, ymax = (
            -self.draw_margin_cells,
            self.space_height - 1 + self.draw_margin_cells,
        )
        number_plane: NumberPlane = NumberPlane(
            x_range=[xmin, xmax + 1, 1],
            y_range=[ymin, ymax + 1, 1],
            x_length=(xmax - xmin + 1) * self.cell_size,
            y_length=(ymax - ymin + 1) * self.cell_size,
            background_line_style={"stroke_opacity": 0.25},
        )
        number_plane.move_to(ORIGIN)
        self.wait()
        self.mapper: GridMapper = GridMapper(
            number_plane=number_plane,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        agent_id2agent_mobject_dic, agent_id2label_dic, agents_group = (
            self._generate_agents()
        )
        self.add(agents_group)
        self.play(
            LaggedStart(
                *[
                    FadeIn(agent_id2agent_mobject_dic[agent_id], scale=0.5)
                    for agent_id in self.agent_ids
                ],
                lag_ratio=0.05,
            ),
            LaggedStart(
                *[
                    FadeIn(agent_id2label_dic[agent_id], scale=0.5)
                    for agent_id in self.agent_ids
                ],
                lag_ratio=0.05,
            ),
            run_time=1.0,
        )
        time_text: Text = Text(self.times[0], font_size=28).to_corner(UL)
        self.add(time_text)
        for time in self.times:
            new_time_text: Text = Text(time, font_size=28).to_corner(UL)
            self.play(Transform(time_text, new_time_text), run_time=0.2)
            anims: list[Mobject] = []
            for agent_id, old_pos, new_pos in self.time2move_dic[time]:
                agent_mobject: SVGMobject | Dot = agent_id2agent_mobject_dic[agent_id]
                old_manim_coord: ndarray = self.mapper.map_pos_to_manim_coord(old_pos)
                current_manim_coord: ndarray = agent_mobject.get_center()
                assert np.allclose(old_manim_coord, current_manim_coord), (
                    f"Agent ID {agent_id} is not at the expected position. "
                    f"Expected: {old_manim_coord}, Actual: {current_manim_coord}"
                )
                new_manim_coord: ndarray = self.mapper.map_pos_to_manim_coord(new_pos)
                label: Text = agent_id2label_dic[agent_id]
                anims.append(agent_mobject.animate.move_to(new_manim_coord))
                anims.append(label.animate.move_to(new_manim_coord + UP * 0.05))
            if anims:
                self.play(AnimationGroup(*anims, lag_ratio=0.0), run_time=0.8)

    def _generate_agents(
        self,
    ) -> tuple[dict[int, SVGMobject | Dot], dict[int, Text], VGroup]:
        agent_id2agent_mobject_dic: dict[int, SVGMobject | Dot] = {}
        agent_id2label_dic: dict[int, Text] = {}
        agents_group: VGroup = VGroup()
        for agent_id in self.agent_ids:
            agent_name: str = self.agent_id2name_dic[agent_id]
            init_pos: Pos = self.agent_id2init_pos_dic[agent_id]
            manim_coord: ndarray = self.mapper.map_pos_to_manim_coord(init_pos)
            svg_path: Optional[str] = self.name2svg_path_dic.get(
                self._match_names(agent_name, list(self.name2svg_path_dic.keys())), None
            )
            agent_manim_obj: SVGMobject | Dot
            if svg_path is not None:
                agent_manim_obj = SVGMobject(svg_path)
            else:
                agent_manim_obj = Dot()
            agent_manim_obj.scale(
                self.name2icon_scale_dic.get(
                    self._match_names(
                        agent_name, list(self.name2icon_scale_dic.keys())
                    ),
                    1.0,
                )
                * self.cell_size
            )
            agent_manim_obj.set_color("WHITE")
            agent_manim_obj.move_to(manim_coord)
            label: Text = Text(agent_name, font_size=18)
            label.next_to(agent_manim_obj, UP, buff=0.05)
            label.add_updater(
                lambda m, dt, d=agent_manim_obj: m.next_to(d, UP, buff=0.05)
            )
            agent_id2agent_mobject_dic[agent_id] = agent_manim_obj
            agent_id2label_dic[agent_id] = label
            agents_group.add(agent_manim_obj, label)
        return agent_id2agent_mobject_dic, agent_id2label_dic, agents_group
