"""
1次元ランジェバン方程式の時空間 (t-x) シミュレーション
テーマ: ピッチフォーク分岐と、それに伴う確率分布の時間発展
"""

import numpy as np

# 明示的なインポート
from manim import (
    BLUE,
    GRAY,
    # 定数・色・配置
    RED,
    RIGHT,
    TEAL,
    UL,
    UP,
    YELLOW,
    Arrow,
    Axes,
    DecimalNumber,
    Dot,
    FadeIn,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    # アニメーション用クラス
    Write,
    config,
    linear,
)


# --- 共通の物理パラメータ・関数 ---
def get_bifurcation_force(x, t, t_bifurcation=3.0):
    """ピッチフォーク分岐の力場計算"""
    # 分岐パラメータ a(t)
    a = 0.8 * (t - t_bifurcation)
    # 力 F(x) = -x^3 + a(t)x
    return 0.8 * (-(x**3) + a * x)


class LangevinSpaceTime(Scene):
    """
    シーン1: 時空図上でのサンプルパスの枝分かれ（一本ずつ描画）
    """

    def construct(self):
        # --- パラメータ ---
        np.random.seed(42)
        n_particles = 10
        t_max = 10.0
        dt = 0.02
        sigma = 0.5
        t_bifurcation = 3.0

        # --- 描画オブジェクト ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=11,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()
        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # 背景ベクトル場
        vector_field = VGroup()
        t_steps = np.linspace(0.5, t_max - 0.5, 18)
        x_steps = np.linspace(-2.5, 2.5, 15)
        for t_val in t_steps:
            for x_val in x_steps:
                f = get_bifurcation_force(x_val, t_val, t_bifurcation)
                vec_t = 0.3
                vec_x = f * 0.15
                color = RED if f > 0 else TEAL
                arrow = Arrow(
                    start=axes.c2p(t_val, x_val),
                    end=axes.c2p(t_val + vec_t, x_val + vec_x),
                    buff=0,
                    max_tip_length_to_length_ratio=0.35,
                    max_stroke_width_to_length_ratio=10,
                    stroke_width=1.5,
                    color=color,
                    stroke_opacity=0.4,
                )
                vector_field.add(arrow)

        title = Text("ピッチフォーク分岐 (サンプルパス)", font_size=36).to_edge(UP)

        self.add(axes, labels)
        self.play(FadeIn(vector_field), Write(title), run_time=1.5)

        # --- シミュレーション ---
        finished_paths = VGroup()
        self.add(finished_paths)

        for i in range(n_particles):
            prog_text = Text(
                f"試行: {i + 1}/{n_particles}", font_size=24, color=YELLOW
            ).to_corner(UL)
            self.add(prog_text)

            t_tracker = ValueTracker(0.0)
            start_x = np.random.normal(0, 0.2)

            dot = Dot(color=YELLOW, radius=0.08)
            dot.move_to(axes.c2p(0, start_x))

            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.8,
                stroke_width=2.0,
                dissipating_time=None,
            )
            self.add(trace, dot)

            dot.sim_t = 0.0
            dot.sim_x = start_x

            def update_particle(mob):
                target_t = t_tracker.get_value()
                current_t = mob.sim_t
                current_x = mob.sim_x

                while current_t < target_t:
                    step_dt = min(dt, target_t - current_t)
                    if step_dt <= 1e-6:
                        break

                    f = get_bifurcation_force(current_x, current_t, t_bifurcation)
                    noise = np.random.normal(0, 1)
                    dx = f * step_dt + sigma * np.sqrt(step_dt) * noise

                    current_x += dx
                    current_t += step_dt

                    if current_x > 3.5:
                        current_x = 3.5
                    if current_x < -3.5:
                        current_x = -3.5

                mob.sim_t = current_t
                mob.sim_x = current_x
                mob.move_to(axes.c2p(current_t, current_x))

            dot.add_updater(update_particle)
            self.play(
                t_tracker.animate.set_value(t_max), run_time=1.5, rate_func=linear
            )

            dot.remove_updater(update_particle)
            dot.set_opacity(0)
            finished_paths.add(trace)
            self.remove(prog_text)

        self.wait(1)


class LangevinDistribution(Scene):
    """
    シーン2: 多数の粒子による分布の時間発展
    時空図 (t-x) 上に、その時刻における分布曲線を重ねて表示する
    """

    def construct(self):
        # --- パラメータ ---
        np.random.seed(999)
        n_particles = 150  # 粒子数
        t_max = 10.0
        dt = 0.05  # 描画更新用ステップ
        sim_dt = 0.01  # 物理計算用サブステップ
        sigma = 0.5
        t_bifurcation = 3.0

        # --- 描画オブジェクト ---
        # 単一の時空図
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()

        labels = axes.get_axis_labels(x_label="t", y_label="x")

        title = Text("確率分布の時間発展 (時空図)", font_size=36).to_edge(UP)

        self.add(axes, labels, title)

        # --- 粒子の初期化 ---
        particles = VGroup()
        particle_data = []  # [current_t, current_x]

        for _ in range(n_particles):
            # 初期位置のバラつき
            start_x = np.random.normal(0, 0.3)

            dot = Dot(color=YELLOW, radius=0.04, fill_opacity=0.5)
            dot.move_to(axes.c2p(0, start_x))

            # 軌跡（薄く残す）
            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.2,  # 薄くして分布曲線を見やすくする
                stroke_width=1.0,
                dissipating_time=3.0,
            )

            particles.add(dot)
            self.add(trace)  # traceは個別にadd
            # dotはparticlesごとaddするため、個別addは不要

            particle_data.append(start_x)

        # 【重要】Updaterを持つVGroup自体をシーンに追加しないとアニメーションしません
        self.add(particles)

        # --- 分布曲線の初期化 ---
        # axes 上に描画する線。
        # 時刻 t の位置に、xに応じた密度分布を「右向きの山」として描画する
        distribution_curve = axes.plot_line_graph(
            x_values=[0, 0],
            y_values=[-3, 3],
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=4,
        )
        self.add(distribution_curve)

        # 分布曲線を目立たせるためのテキスト
        dist_label = Text("P(x, t)", font_size=24, color=BLUE).next_to(
            distribution_curve, UP
        )
        self.add(dist_label)

        # --- シミュレーション用トラッカー ---
        t_tracker = ValueTracker(0.0)
        time_label = DecimalNumber(0, num_decimal_places=1, include_sign=False).next_to(
            title, RIGHT
        )
        self.add(time_label)

        # --- アップデータ ---
        def update_scene(mob):
            # 現在のアニメーション時刻
            current_t = t_tracker.get_value()
            time_label.set_value(current_t)

            # 1. 粒子の物理計算と移動
            steps = int(dt / sim_dt)
            for _ in range(steps):
                if current_t >= t_max:
                    break

                x_vals = np.array(particle_data)

                # 力の計算
                f = get_bifurcation_force(x_vals, current_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)

                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                x_vals += dx

                # 境界条件
                x_vals = np.clip(x_vals, -3.5, 3.5)

                # データ更新
                for i in range(n_particles):
                    particle_data[i] = x_vals[i]

            # ドットの描画位置更新
            for i, dot in enumerate(particles):
                dot.move_to(axes.c2p(current_t, particle_data[i]))

            # 2. 分布の計算と描画 (同じグラフ上に重ねる)
            x_grid = np.linspace(-3, 3, 80)
            density = np.zeros_like(x_grid)

            # KDE
            bandwidth = 0.3
            current_x_vals = np.array(particle_data)

            for x_p in current_x_vals:
                density += np.exp(-0.5 * ((x_grid - x_p) / bandwidth) ** 2)

            # 正規化とスケーリング
            # densityの高さを時間軸(t)上の幅に変換する
            scale_factor = 1.0  # 密度1あたりのt軸上の長さ
            density = (
                density / (n_particles * bandwidth * np.sqrt(2 * np.pi)) * 3.0
            )  # 高さを強調

            # 時刻 t を基準線として、密度分だけ右にずらす
            # x_values は t座標, y_values は x座標
            t_coords = current_t + density * scale_factor

            new_curve = axes.plot_line_graph(
                x_values=t_coords,
                y_values=x_grid,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=4,
            )
            distribution_curve.become(new_curve)

            # ラベルも追従させる
            dist_label.next_to(axes.c2p(current_t, 3.2), UP)

        particles.add_updater(update_scene)

        self.play(t_tracker.animate.set_value(t_max), run_time=8.0, rate_func=linear)

        particles.remove_updater(update_scene)
        self.wait(1)


if __name__ == "__main__":
    config.quality = "medium_quality"
    scene = LangevinDistribution()
    scene.render()
