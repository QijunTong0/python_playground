import numpy as np
from manim import (
    BLUE,
    DOWN,
    UP,
    WHITE,
    YELLOW,
    Axes,
    Create,
    Dot,
    FadeIn,
    MoveAlongPath,
    Scene,
    Text,
    VMobject,
    linear,
)


class SDESimulation(Scene):
    def construct(self):
        # --- 1. シミュレーション設定 ---
        # 厳密シミュレーションなので dt が大きくても統計的に正しいが、
        # 2次元の軌跡を滑らかに見せるために細かめに設定する。
        dt = 0.05
        total_time = 15.0
        steps = int(total_time / dt)

        # 座標系の設定 (2次元空間 X1 vs X2)
        # 積分ブラウン運動は拡散が早いため(t^3)、範囲を広めに取る
        ax = Axes(
            x_range=[-8, 8, 2],
            y_range=[-6, 6, 2],
            axis_config={"include_tip": True},
            x_length=7,
            y_length=7,
        ).add_coordinates()

        labels = ax.get_axis_labels(x_label="X_1", y_label="X_2")
        title = Text(
            "2D Integrated Brownian Motion (Exact Simulation)", font_size=32
        ).to_edge(UP)

        self.add(ax, labels, title)

        # --- 2. データ生成 ---
        # 状態: [位置, 速度]
        # 2つの独立した次元 (Dim 1, Dim 2) を用意
        # 初期値: 原点停止
        state_1 = np.array([0.0, 0.0])  # X1, V1
        state_2 = np.array([0.0, 0.0])  # X2, V2

        # 描画用の点リスト（Manimの座標系に変換済み）
        path_points = [ax.c2p(state_1[0], state_2[0])]

        np.random.seed(99)  # 画面内に収まりつつ、特徴的な動きをするシード

        for _ in range(steps):
            # 共通の共分散行列 (各次元独立)
            # Q_dt = [[ dt^3/3, dt^2/2 ],
            #         [ dt^2/2, dt     ]]
            cov = np.array([[(1 / 3) * dt**3, (1 / 2) * dt**2], [(1 / 2) * dt**2, dt]])

            # --- 次元 1 の更新 ---
            # 平均: E[Z_{t+dt}] = [X + V*dt, V]
            mean_1 = np.array([state_1[0] + state_1[1] * dt, state_1[1]])
            state_1 = np.random.multivariate_normal(mean_1, cov)

            # --- 次元 2 の更新 ---
            mean_2 = np.array([state_2[0] + state_2[1] * dt, state_2[1]])
            state_2 = np.random.multivariate_normal(mean_2, cov)

            # 位置 (X1, X2) を保存
            path_points.append(ax.c2p(state_1[0], state_2[0]))

        # --- 3. アニメーション要素の作成 ---

        # 軌跡オブジェクト (VMobjectを使って滑らかな線にする)
        path = VMobject()
        path.set_points_as_corners(path_points)
        path.set_color(BLUE)
        path.set_stroke(width=3)

        # 先端の粒子
        dot = Dot(color=YELLOW, radius=0.08)
        dot.move_to(path_points[0])

        # --- 4. 描画実行 ---

        self.add(dot)  # 粒子を最初に表示

        # 軌跡の描画と粒子の移動を同期
        self.play(Create(path), MoveAlongPath(dot, path), run_time=6, rate_func=linear)

        # 解説テキスト
        info = Text(
            "Trajectory of particle driven by random acceleration (White Noise).\nSmooth path (differentiable velocity) unlike standard Brownian motion.",
            font_size=20,
            color=WHITE,
            line_spacing=1.5,
        ).to_edge(DOWN)

        self.play(FadeIn(info))
        self.wait(2)
