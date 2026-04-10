from collections import defaultdict
from typing import Any, Optional

import numpy as np
from manim import (
    WHITE,
    Animation,
    AnimationGroup,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Mobject,
    Rectangle,
    SVGMobject,
    Text,
    Transform,
    VGroup,
    config as manim_config,
)
from manim.constants import DOWN, LEFT, RIGHT, UL, UP
from numpy import ndarray

from .base import Animator


class SocialNetworkAnimator(Animator):
    def _get_log_configs(self, config: dict[str, Any]):
        super()._get_log_configs(config)
        self.max_tweets_in_tl: int = self.animation_config.get("maxTweetsInTL", 5)
        self.fast_rendering: bool = self.animation_config.get("fastRendering", False)

    def _get_time2social_interactions_dic(
        self,
    ) -> tuple[list[int | str], dict[int, list[dict[str, Any]]]]:
        time2social_interactions_dic: dict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        times: list[int | str] = []
        for log_dic in self.log_dics:
            if log_dic.get("type") in [
                "follow",
                "unfollow",
                "tweet",
            ]:
                time: int | str = log_dic["time"]
                if time not in times:
                    times.append(time)
                time2social_interactions_dic[time].append(log_dic)
        return times, time2social_interactions_dic

    def _make_social_layout(self, center: ndarray, radius: float) -> dict[int, ndarray]:
        num_agents: int = len(self.agent_ids)
        agent_id2manim_coord: dict[int, ndarray] = {}
        for i, agent_id in enumerate(self.agent_ids):
            angle: float = 2 * np.pi * i / num_agents
            manim_coord: ndarray = center + radius * np.array(
                [np.cos(angle), np.sin(angle), 0.0]
            )
            agent_id2manim_coord[agent_id] = manim_coord
        return agent_id2manim_coord

    def construct(self) -> None:
        self.agent_id2name_dic: dict[int, str] = self._get_agent_id2name_dic()
        self.times, self.time2social_interactions_dic = (
            self._get_time2social_interactions_dic()
        )
        self.times: list[int | str] = sorted(self.times)
        self.agent_ids: list[int] = sorted(list(self.agent_id2name_dic.keys()))
        time_text = Text(self.times[0], font_size=28).to_corner(UL)
        self.add(time_text)
        frame_w: int = manim_config.frame_width
        sns_w: float = frame_w * 0.5
        frame_h: int = manim_config.frame_height
        sns_h: float = frame_h
        sns_area: Rectangle = Rectangle(width=sns_w, height=sns_h, stroke_opacity=0.0)
        sns_area.to_edge(LEFT)
        tl_area: Rectangle = Rectangle(
            width=frame_w - sns_w, height=frame_h, stroke_opacity=0.0
        )
        tl_area.to_edge(RIGHT)
        social_layout: dict[int, ndarray] = self._make_social_layout(
            center=sns_area.get_center() + LEFT * 0.3,
            radius=min(sns_w, sns_h) * 0.35,
        )
        (
            self.agent_id2agent_mobject_dic,
            agent_id2label_dic,
            self.agent_id2group_dic,
            agents_group,
        ) = self._generate_agents(social_layout=social_layout)
        self.play(
            LaggedStart(
                *[
                    FadeIn(self.agent_id2group_dic[agent_id], scale=0.5)
                    for agent_id in self.agent_ids
                ],
                lag_ratio=0.05,
            ),
            run_time=1.0,
        )
        tl_box: Rectangle = Rectangle(
            width=tl_area.width * 0.95, height=tl_area.height * 0.95, stroke_opacity=0.4
        )
        tl_box.move_to(tl_area.get_center())
        tl_title: Text = Text("Latest Tweets", font_size=20)
        tl_title.next_to(tl_box.get_top(), DOWN, buff=0.15).align_to(
            tl_box.get_left(), LEFT
        ).shift(RIGHT * 0.15)
        self.add(tl_box, tl_title)
        self.wait(0.1)
        self.latest_tweets: Optional[list[Text]] = None
        self.tl_group: Optional[VGroup] = None
        if self.fast_rendering:
            self.latest_tweets = []
        else:
            self.tl_group = VGroup()
            self.tl_group.is_tl = True
            self.add(self.tl_group)
        follow_edge_dic: dict[tuple[int, int], Arrow] = {}
        for time in self.times:
            new_time_text: Text = Text(time, font_size=28).to_corner(UL)
            self.play(Transform(time_text, new_time_text), run_time=0.2)
            anims: list[Mobject] = []
            for log_dic in self.time2social_interactions_dic.get(time, []):
                if log_dic["type"] == "tweet":
                    self._update_tl(
                        tl_box=tl_box,
                        log_dic=log_dic,
                    )
                elif log_dic["type"] in ["follow", "unfollow"]:
                    self._update_edges(
                        anims=anims,
                        log_dic=log_dic,
                        is_follow=(log_dic["type"] == "follow"),
                        follow_edge_dic=follow_edge_dic,
                    )
                else:
                    raise ValueError(
                        f"Unknown social interaction type: {log_dic['type']}"
                    )
            if len(anims) > 0:
                self.play(AnimationGroup(*anims, lag_ratio=0.05), run_time=0.6)

    def _generate_agents(
        self, social_layout: dict[int, ndarray]
    ) -> tuple[dict[int, SVGMobject | Dot], dict[int, Text], dict[int, VGroup], VGroup]:
        agent_id2agent_mobject_dic: dict[int, SVGMobject | Dot] = {}
        agent_id2label_dic: dict[int, Text] = {}
        agent_id2group_dic: dict[int, VGroup] = {}
        agents_group: VGroup = VGroup()
        for agent_id in self.agent_ids:
            agent_name: str = self.agent_id2name_dic[agent_id]
            pos: ndarray = social_layout[agent_id]
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
                * 0.4
            )
            agent_manim_obj.set_color("WHITE")
            agent_manim_obj.move_to(pos)
            label: Text = Text(agent_name, font_size=7)
            label.add_updater(lambda m, d=agent_manim_obj: m.next_to(d, UP, buff=0.05))
            label.set_z_index(3)
            label.update()
            agent_group = VGroup(agent_manim_obj, label)
            agent_id2agent_mobject_dic[agent_id] = agent_manim_obj
            agent_id2label_dic[agent_id] = label
            agent_id2group_dic[agent_id] = agent_group
            agents_group.add(agent_group)
        return (
            agent_id2agent_mobject_dic,
            agent_id2label_dic,
            agent_id2group_dic,
            agents_group,
        )

    def _update_edges(
        self,
        anims: list[Mobject],
        log_dic: dict[str, Any],
        is_follow: bool,
        follow_edge_dic: dict[tuple[int, int], Arrow],
    ) -> None:
        src_agent_id: int = int(log_dic["agent_id"])
        dst_agent_id: int = int(log_dic["target_agent_id"])
        edge_key: tuple[int, int] = (src_agent_id, dst_agent_id)
        edge: Arrow
        if is_follow:
            if edge_key not in follow_edge_dic:
                edge = self._make_edge(src_agent_id, dst_agent_id)
                anims.append(Create(edge))
                follow_edge_dic[edge_key] = edge
                anims.append(
                    Indicate(
                        self.agent_id2group_dic[src_agent_id],
                        scale_factor=1.3,
                        run_time=0.3,
                    )
                )
                anims.append(
                    Indicate(
                        self.agent_id2group_dic[dst_agent_id],
                        scale_factor=1.3,
                        run_time=0.3,
                    )
                )
        else:
            if edge_key in follow_edge_dic:
                edge = follow_edge_dic.pop(edge_key)
                anims.append(FadeOut(edge))

    def _make_edge(self, src_agent_id: int, dst_agent_id: int) -> Arrow:
        edge: Arrow = Arrow(
            start=self.agent_id2agent_mobject_dic[src_agent_id].get_center(),
            end=self.agent_id2agent_mobject_dic[dst_agent_id].get_center(),
            buff=0.12,
            stroke_width=1.5,
            color=WHITE,
            max_tip_length_to_length_ratio=0.05,
        )
        edge.set_z_index(1)
        return edge

    def _update_tl(
        self,
        tl_box: Rectangle,
        log_dic: dict[str, Any],
    ) -> None:
        anchor = tl_box.get_top() + DOWN * 0.65 + RIGHT * 0.2
        agent_id: int = int(log_dic["agent_id"])
        message: str = log_dic["message"]
        if len(message) > 78:
            message = message[:75] + "..."
        agent_name: str = self.agent_id2name_dic[agent_id]
        tweet_text: Text = Text(f"{agent_name}:\n {message}", font_size=12)
        if self.fast_rendering:
            assert self.latest_tweets is not None
            self.latest_tweets = [tweet_text] + self.latest_tweets
            if len(self.latest_tweets) > self.max_tweets_in_tl:
                self.latest_tweets.pop()
            for i, tl in enumerate(self.latest_tweets):
                tl.move_to(anchor + DOWN * 0.5 * i).align_to(tl_box, LEFT).shift(
                    RIGHT * 0.2
                )
            self.remove(*[m for m in self.mobjects if getattr(m, "is_tl", False)])
            for tl in self.latest_tweets:
                tl.is_tl = True
                self.add(tl)
        else:
            assert self.tl_group is not None
            tweet_text.is_tl = True
            tweet_text.move_to(anchor + UP * 0.15).align_to(tl_box, LEFT).shift(
                RIGHT * 0.2
            )
            new_group: VGroup = VGroup(tweet_text, *self.tl_group.submobjects)
            new_group.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
            new_group.move_to(anchor, aligned_edge=UP)
            new_group.align_to(tl_box, LEFT).shift(RIGHT * 0.2)
            if len(new_group) > self.max_tweets_in_tl:
                new_group = VGroup(*new_group[: self.max_tweets_in_tl])
            old_subs = list(self.tl_group.submobjects)
            new_subs = list(new_group.submobjects)
            tweet_anims: list[Animation] = []
            for i in range(min(len(old_subs), len(new_subs) - 1)):
                tweet_anims.append(
                    old_subs[i].animate.move_to(new_subs[i + 1].get_center())
                )
            tweet_anims.append(FadeIn(tweet_text, shift=DOWN * 0.15))
            for m in old_subs[self.max_tweets_in_tl - 1 :]:
                if m not in new_subs:
                    tweet_anims.append(FadeOut(m))
            self.play(AnimationGroup(*tweet_anims, lag_ratio=0.0), run_time=0.1)
            self.tl_group.become(new_group)
