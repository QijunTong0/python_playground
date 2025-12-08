"""
1次元ランジェバン方程式の時空間 (t-x) シミュレーション
背景: 時空平面全体におけるドリフトベクトル場（決定論的な流れ）
前景: その場の中を進む確率的なサンプルパス
"""

import numpy as np

# 明示的なインポート
from manim import (
    DOWN,
    GRAY,
    # 定数・色・配置
    RED,
    TEAL,
    UP,
    YELLOW,
    Arrow,
    Axes,
    # 追記: アニメーション用クラスのインポート
    Create,
    Dot,
    FadeIn,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    Write,
    config,
)


class LangevinSpaceTime(Scene):
    def construct(self):
        # --- 1. パラメータ設定 ---
        np.random.seed(42)
        n_particles = 8  # サンプルパスの本数
        t_max = 10.0  # シミュレーション終了時間（横軸の最大値）
        dt = 0.05  # シミュレーションのタイムステップ
        sigma = 0.5  # ノイズの強さ (拡散係数)

        # ポテンシャルパラメータ U(x, t)
        # 波のように移動するポテンシャルを考える
        # F(x, t) = -dU/dx
        # ここでは単純に正弦波の谷が斜めに走る状況を作る
        # U(x, t) ~ cos(k*x - w*t)
        wave_k = 1.5
        wave_w = 1.0

        # --- 2. 物理関数の定義 ---
        def force(x: float, t: float) -> float:
            """
            時刻 t, 位置 x における x方向の力 (ドリフト項)
            F = -dU/dx
            U = -sin(k*x - w*t) とすると (谷に落ちる)
            F = k * cos(k*x - w*t)
            さらに原点付近に留めるためのバネ項 -0.1*x を追加
            """
            return 1.5 * wave_k * x**2 - wave_w * t - 0.1 * x

        # --- 3. 描画オブジェクト: 座標軸 (t-x 平面) ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=11,
            y_length=6,
            axis_config={
                "include_tip": True,
                # "tip_shape": Arrow,  # 削除: これがTypeErrorの原因でした
                "color": GRAY,
            },
        ).add_coordinates()

        labels = axes.get_axis_labels(x_label="t", y_label="x")

        self.add(axes, labels)

        # --- 4. 背景ベクトル場の描画 (Space-Time Flow) ---
        # 時空全体にわたって、粒子が従うべき「流れ」を矢印で描く
        # ベクトル v = (dt, dx) ~ (1, force)
        vector_field = VGroup()

        # グリッド生成
        t_steps = np.linspace(0.5, t_max - 0.5, 12)
        x_steps = np.linspace(-2.5, 2.5, 10)

        for t_val in t_steps:
            for x_val in x_steps:
                f = force(x_val, t_val)

                # 時空上のベクトル: (時間方向の進み, 位置方向の力)
                # 視覚的にわかりやすくするため、成分を調整
                vec_t = 0.8  # 横向きの成分（固定）
                vec_x = f * 0.4  # 縦向きの成分（力に比例）

                # 開始点
                start_point = axes.c2p(t_val, x_val)
                # 終了点
                end_point = axes.c2p(t_val + vec_t, x_val + vec_x)

                # 力の大きさで色を変える
                color = RED if f > 0 else TEAL
                # 力が強い場所（急な坂）ほど透明度を上げて目立たせる
                opacity = min(abs(f) * 0.4 + 0.2, 0.8)

                arrow = Arrow(
                    start=start_point,
                    end=end_point,
                    buff=0,
                    max_tip_length_to_length_ratio=0.25,
                    stroke_width=2,
                    color=color,
                    stroke_opacity=opacity,
                )
                vector_field.add(arrow)

        title = Text("Langevin Dynamics in Space-Time (t, x)", font_size=32)
        title.to_edge(UP)
        subtitle = Text(
            "Arrows indicate drift field: force + time evolution",
            font_size=20,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN)

        self.add(vector_field, title, subtitle)

        # 背景のアニメーション（フェードイン）
        self.play(
            *(
                [
                    Create(axes),
                    Write(labels),
                    FadeIn(vector_field),
                    Write(title),
                    Write(subtitle),
                ]
            ),
            run_time=2,
        )

        # --- 5. サンプルパスのシミュレーション ---

        # 時間管理用のValueTracker
        # これはアニメーションの進行（描画上の時間）を管理するもので、
        # 物理シミュレーションの t と同期させる
        t_tracker = ValueTracker(0)

        particles = VGroup()
        traces = VGroup()

        # 粒子の初期化
        dot_objects = []
        particle_data = []  # 現在の (t, x) を保持

        for i in range(n_particles):
            # 初期位置 x を分散させる
            start_x = np.random.uniform(-2.0, 2.0)

            dot = Dot(color=YELLOW, radius=0.08)
            dot.move_to(axes.c2p(0, start_x))  # t=0

            # 軌跡
            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.6,
                stroke_width=2,
                dissipating_time=None,  # 軌跡を消さない
            )

            particles.add(dot)
            traces.add(trace)
            dot_objects.append(dot)
            # データ: [current_t, current_x]
            particle_data.append([0.0, start_x])

        self.add(traces, particles)

        # アニメーション更新関数
        def update_simulation(mob):
            # 引数 dt_frame を削除しました。
            # 物理シミュレーション用の dt (0.05) は外部スコープの変数を使用します。

            # 全粒子を少しずつ進める
            steps_per_frame = 2  # フレームごとの計算精度向上のためのサブステップ
            sim_dt = dt / steps_per_frame

            for _ in range(steps_per_frame):
                for i, dot in enumerate(dot_objects):
                    curr_t, curr_x = particle_data[i]

                    if curr_t >= t_max:
                        continue  # 終了した粒子は止める

                    # ランジェバン方程式
                    # dX = F(X, t)dt + sigma * dW
                    f = force(curr_x, curr_t)
                    noise = np.random.normal(0, 1)

                    dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise

                    new_t = curr_t + sim_dt
                    new_x = curr_x + dx

                    # 画面外制限
                    if new_x > 3.0:
                        new_x = 3.0
                    if new_x < -3.0:
                        new_x = -3.0

                    particle_data[i] = [new_t, new_x]

                    # 描画位置更新
                    dot.move_to(axes.c2p(new_t, new_x))

        # シミュレーション開始
        particles.add_updater(update_simulation)

        # 時間を進める (アニメーション時間として10秒かける)
        self.wait(10)

        particles.remove_updater(update_simulation)
        self.wait(1)


if __name__ == "__main__":
    config.quality = "medium_quality"
    scene = LangevinSpaceTime()
    scene.render()
