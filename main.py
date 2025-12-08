"""
1次元ランジェバン方程式の時空間 (t-x) シミュレーション
テーマ: 経路の分岐と、それに伴う確率分布の時間発展
"""

import numpy as np

# 明示的なインポート
from manim import (
    BLUE,
    GRAY,
    # 定数・色・配置
    RED,
    TEAL,
    UL,
    UP,
    UR,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    # アニメーション用クラス
    Create,
    DashedVMobject,  # 点線用
    DecimalNumber,
    Dot,
    FadeIn,
    MathTex,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    VMobject,  # 修正: VMobjectのインポート漏れを追加
    Write,
    linear,
)


# --- 共通の物理パラメータ・関数 ---
def get_bifurcation_force(x, t, t_bifurcation=3.0):
    """分岐の力場計算"""
    # 分岐パラメータ a(t)
    a = 0.8 * (t - t_bifurcation)
    # 力 F(x) = -x^3 + a(t)x
    return 0.8 * (-(x**3) + a * x)


def compute_kde_distribution(particle_positions, bandwidth=0.3, num_points=80):
    """カーネル密度推定(KDE)による分布計算"""
    x_grid = np.linspace(-3, 3, num_points)
    density = np.zeros_like(x_grid)
    n = len(particle_positions)

    for x_p in particle_positions:
        density += np.exp(-0.5 * ((x_grid - x_p) / bandwidth) ** 2)

    # 正規化とスケーリング (高さ調整)
    density = density / (n * bandwidth * np.sqrt(2 * np.pi)) * 3.0
    return x_grid, density


class LangevinSpaceTime(Scene):
    """
    シーン1: 時空図上でのサンプルパスの分岐（一本ずつ描画）
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

        # 修正: タイトル削除
        # タイトルやグラフをフェードインで表示
        self.play(FadeIn(axes), Write(labels), run_time=1.5)
        self.play(FadeIn(vector_field), run_time=1.0)

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

        # 修正: タイトル削除
        # アニメーションで表示
        self.play(FadeIn(axes), Write(labels), run_time=1.5)

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
                dissipating_time=None,
            )

            particles.add(dot)
            self.add(trace)  # traceは個別にadd
            # dotはparticlesごとaddするため、個別addは不要

            particle_data.append(start_x)

        # 粒子は最初は見えない状態からFadeInさせる
        self.play(FadeIn(particles), run_time=1.0)

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

        # MathTexを使用してLaTeXレンダリング
        dist_label = MathTex(r"P(x, t)", font_size=24, color=BLUE).next_to(
            distribution_curve, UP
        )

        # 分布曲線とラベルもアニメーションで表示
        self.play(Create(distribution_curve), Write(dist_label), run_time=1.0)

        # --- シミュレーション用トラッカー ---
        t_tracker = ValueTracker(0.0)
        # 修正: タイトルがないので、位置を右上に固定
        time_label = DecimalNumber(
            0, num_decimal_places=1, include_sign=False
        ).to_corner(UR)
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
            # 共通化したKDE関数を使用
            x_grid, density = compute_kde_distribution(particle_data)

            # スケーリング
            scale_factor = 1.0
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


class LangevinInverseProblem(Scene):
    """
    シーン3: 逆問題デモンストレーション
    1. 未来の周辺分布(t=2,4,6,8)を「観測データ」として事前に描画
    2. t=0からシミュレーションを行い、分布が観測データに一致していく様子を見せる
    """

    def construct(self):
        # --- パラメータ (Scene2と同じにして整合性を取る) ---
        seed = 999
        n_particles = 150
        t_max = 10.0
        dt = 0.05
        sim_dt = 0.01
        sigma = 0.5
        t_bifurcation = 3.0

        target_times = [2.0, 4.0, 6.0, 8.0]  # 分布を表示する時刻

        # --- 描画オブジェクト ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()
        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # 修正: タイトルとサブタイトルを削除
        self.play(FadeIn(axes), Write(labels), run_time=1.5)

        # --- 1. 事前計算 (Target Distributions) ---
        # シミュレーションをバックグラウンドで回して、ターゲット時刻の分布を計算する
        np.random.seed(seed)  # 同じシードを使用

        # 粒子初期化（計算用）
        sim_particles = np.random.normal(0, 0.3, n_particles)

        target_curves = VGroup()

        current_sim_t = 0.0
        target_idx = 0

        # ステップごとのループ（描画はしない）
        while current_sim_t < t_max and target_idx < len(target_times):
            # 次のターゲット時刻までのステップ数
            next_target_t = target_times[target_idx]

            # 到達していないなら進める
            while current_sim_t < next_target_t:
                f = get_bifurcation_force(sim_particles, current_sim_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)
                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                sim_particles += dx
                sim_particles = np.clip(sim_particles, -3.5, 3.5)
                current_sim_t += sim_dt

            # ターゲット時刻に到達したので分布を作成
            x_grid, density = compute_kde_distribution(sim_particles)

            # 分布曲線の生成 (点線、薄い色)
            scale_factor = 1.0
            t_coords = next_target_t + density * scale_factor

            # VMobjectを作成
            points = [axes.c2p(t, x) for t, x in zip(t_coords, x_grid)]

            # 1本の連続した線として作成
            curve_vm = VMobject()
            curve_vm.set_points_as_corners(points)
            curve_vm.set_color(GRAY)

            # DashedVMobjectで点線化
            dashed_curve = DashedVMobject(curve_vm, num_dashes=25, dashed_ratio=0.5)

            # ラベル
            label = MathTex(
                f"t={int(next_target_t)}", font_size=20, color=GRAY
            ).next_to(axes.c2p(next_target_t, 3.2), UP)

            target_curves.add(dashed_curve, label)
            target_idx += 1

        # ターゲット分布を表示
        self.play(Create(target_curves), run_time=2.0)
        self.wait(0.5)

        # --- 2. リアルタイムシミュレーション (Animation) ---
        # シードをリセットして、同じ挙動を再現
        np.random.seed(seed)

        particles = VGroup()
        particle_data = []

        # 粒子の初期化 (描画用)
        for _ in range(n_particles):
            start_x = np.random.normal(0, 0.3)
            dot = Dot(color=YELLOW, radius=0.04, fill_opacity=0.5)
            dot.move_to(axes.c2p(0, start_x))

            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.2,
                stroke_width=1.0,
                dissipating_time=None,
            )
            particles.add(dot)
            self.add(trace)
            particle_data.append(start_x)

        self.play(FadeIn(particles), run_time=1.0)

        # リアルタイム分布曲線
        realtime_curve = axes.plot_line_graph(
            x_values=[0, 0],
            y_values=[-3, 3],
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=4,
        )
        dist_label = MathTex(r"P_{sim}(x, t)", font_size=24, color=BLUE).next_to(
            realtime_curve, UP
        )

        self.play(Create(realtime_curve), Write(dist_label), run_time=1.0)

        # トラッカー
        t_tracker = ValueTracker(0.0)
        # 修正: タイトルがないので、位置を右上に固定
        time_label = DecimalNumber(
            0, num_decimal_places=1, include_sign=False
        ).to_corner(UR)
        self.add(time_label)

        def update_scene(mob):
            current_t = t_tracker.get_value()
            time_label.set_value(current_t)

            # 物理計算
            steps = int(dt / sim_dt)
            for _ in range(steps):
                if current_t >= t_max:
                    break
                x_vals = np.array(particle_data)
                f = get_bifurcation_force(x_vals, current_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)
                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                x_vals += dx
                x_vals = np.clip(x_vals, -3.5, 3.5)
                for i in range(n_particles):
                    particle_data[i] = x_vals[i]

            for i, dot in enumerate(particles):
                dot.move_to(axes.c2p(current_t, particle_data[i]))

            # 分布更新
            x_grid, density = compute_kde_distribution(particle_data)
            scale_factor = 1.0
            t_coords = current_t + density * scale_factor

            new_curve = axes.plot_line_graph(
                x_values=t_coords,
                y_values=x_grid,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=4,
            )
            realtime_curve.become(new_curve)
            dist_label.next_to(axes.c2p(current_t, 3.2), UP)

        particles.add_updater(update_scene)

        self.play(t_tracker.animate.set_value(t_max), run_time=8.0, rate_func=linear)

        particles.remove_updater(update_scene)
        self.wait(1)


def generate_brownian_bridge(start_val, end_val, steps=100, sigma=0.8):
    """始点と終点を結ぶブラウン橋のパスを生成"""
    dt = 1.0 / steps
    t_vals = np.linspace(0, 1, steps + 1)

    # 1. 標準ブラウン運動 W(t) を生成 (W(0)=0)
    noise = np.random.normal(0, np.sqrt(dt), steps)
    w_process = np.concatenate(([0], np.cumsum(noise)))

    # 2. 終端条件を満たすように変換 (tは0~1正規化されている前提)
    # B(t) = x0 + W(t) - t * (W(1) - (x1 - x0))
    # これにより B(0)=x0, B(1)=x1 となる
    bridge = start_val + w_process - t_vals * (w_process[-1] - (end_val - start_val))

    # ノイズ強度 sigma を W(t) の部分に掛けるイメージで調整
    # ただし単純なスケーリングだと端点がずれるため、
    # 実際には W(t) 生成時に sigma を反映させるのが正しいが、
    # 上記式は W(t) が標準(sigma=1)前提の変換式。
    # 任意の sigma に対応させるには、noise 生成時に sigma を掛け、
    # 変換式自体は線形なのでそのままで成立する。

    noise = np.random.normal(0, sigma * np.sqrt(dt), steps)
    w_process = np.concatenate(([0], np.cumsum(noise)))
    bridge = start_val + w_process - t_vals * (w_process[-1] - (end_val - start_val))

    return t_vals, bridge


def run_brownian_bridge_logic(scene):
    """シーン4: ブラウン橋 (Brownian Bridge) デモ"""
    np.random.seed(1234)  # 再現性のため

    # 1. 座標軸 (t: 0->1 を大きく表示)
    axes = Axes(
        x_range=[-0.1, 1.2, 0.2],
        y_range=[-2.5, 2.5, 0.5],
        x_length=10,
        y_length=6,
        axis_config={"include_tip": False, "color": GRAY},
    ).add_coordinates()

    labels = axes.get_axis_labels(x_label="t", y_label="x")

    # タイトル削除（要望通り）
    scene.play(FadeIn(axes), Write(labels), run_time=1.5)

    # 2. 点のプロット
    # t=0 に4点
    start_points_group = VGroup()
    start_dots = []
    start_y_values = np.linspace(-1.5, 1.5, 4) + np.random.normal(0, 0.2, 4)

    for y in start_y_values:
        dot = Dot(point=axes.c2p(0, y), color=BLUE, radius=0.1)
        start_dots.append(dot)
        start_points_group.add(dot)

    # t=1 に5点
    end_points_group = VGroup()
    end_dots = []
    end_y_values = np.linspace(-1.8, 1.8, 5) + np.random.normal(0, 0.2, 5)

    for y in end_y_values:
        dot = Dot(point=axes.c2p(1, y), color=RED, radius=0.1)
        end_dots.append(dot)
        end_points_group.add(dot)

    scene.play(FadeIn(start_points_group), FadeIn(end_points_group), run_time=1.0)

    # テキスト補助
    t0_label = Text("t=0", font_size=24, color=BLUE).next_to(axes.c2p(0, 2.5), UP)
    t1_label = Text("t=1", font_size=24, color=RED).next_to(axes.c2p(1, 2.5), UP)
    scene.play(Write(t0_label), Write(t1_label), run_time=0.5)

    # 3. ランダムに結ぶループ (10回)
    # 生成したパスを保存しておくグループ
    bridges = VGroup()
    scene.add(bridges)

    for i in range(10):
        # ランダム選択
        s_idx = np.random.randint(0, len(start_dots))
        e_idx = np.random.randint(0, len(end_dots))

        s_dot = start_dots[s_idx]
        e_dot = end_dots[e_idx]

        # ハイライト (拡大 & 色変更)
        scene.play(
            s_dot.animate.scale(1.5).set_color(YELLOW),
            e_dot.animate.scale(1.5).set_color(YELLOW),
            run_time=0.2,
        )

        # パス計算
        start_val = start_y_values[s_idx]
        end_val = end_y_values[e_idx]
        t_vals, x_vals = generate_brownian_bridge(
            start_val, end_val, steps=200, sigma=1.2
        )

        # パス描画用VMobject作成
        bridge_path = VMobject()
        points = [axes.c2p(t, x) for t, x in zip(t_vals, x_vals)]
        bridge_path.set_points_as_corners(points)
        bridge_path.set_color(YELLOW)
        bridge_path.set_stroke(width=2, opacity=0.8)

        # アニメーション: Create で左から右へ描画
        scene.play(Create(bridge_path), run_time=0.8, rate_func=linear)

        # 描画完了後、パスを薄くして残す
        bridges.add(bridge_path)
        bridge_path.set_color(WHITE).set_stroke(opacity=0.3, width=1)

        # ドットを元に戻す
        scene.play(
            s_dot.animate.scale(1 / 1.5).set_color(BLUE),
            e_dot.animate.scale(1 / 1.5).set_color(RED),
            run_time=0.2,
        )

    scene.wait(1)


def run_trajectory_inference_logic(scene):
    """シーン4: Brownian Bridgeによるパス生成"""
    np.random.seed(123)
    n_paths = 10
    t_points = [0, 0.25, 0.5, 0.75, 1.0]
    points_per_t = 4

    # 軸の設定 (t: 0-1.2, x: -3-3)
    axes = Axes(
        x_range=[0, 1.2, 0.25],
        y_range=[-3, 3, 1],
        x_length=10,
        y_length=6,
        axis_config={"include_tip": True, "color": GRAY},
    ).add_coordinates()
    labels = axes.get_axis_labels(x_label="t", y_label="x")

    scene.play(FadeIn(axes), Write(labels), run_time=1.5)

    # --- 1. ランダムな点を打つ ---
    all_dots = VGroup()  # 全ての点のグループ
    dots_by_t = []  # 時刻ごとのDotリストのリスト [[Dot, Dot...], [Dot...]]

    for t in t_points:
        dots_at_t = []
        # x座標をランダム生成
        x_coords = np.random.uniform(-2.5, 2.5, points_per_t)
        for x in x_coords:
            dot = Dot(color=WHITE, radius=0.06)
            dot.move_to(axes.c2p(t, x))
            all_dots.add(dot)
            dots_at_t.append(dot)
        dots_by_t.append(dots_at_t)

    scene.play(FadeIn(all_dots), run_time=1.0)

    # --- 2. パス生成ループ ---
    finished_paths = VGroup()
    scene.add(finished_paths)

    for i in range(n_paths):
        # 進行状況表示
        prog_text = Text(
            f"Path: {i + 1}/{n_paths}", font_size=24, color=YELLOW
        ).to_corner(UL)
        scene.add(prog_text)

        # 始点 (t=0) をランダムに選ぶ
        current_dot = np.random.choice(dots_by_t[0])
        scene.play(Flash(current_dot, color=YELLOW, flash_radius=0.2), run_time=0.3)

        # 現在のパスのセグメントを保存するリスト
        current_path_segments = VGroup()

        # t=0 -> 0.25 -> 0.5 ... と繋いでいく
        for j in range(len(t_points) - 1):
            t_start = t_points[j]
            t_end = t_points[j + 1]

            # 始点の座標 (axes座標系から値に戻すのは面倒なので保持しておいたほうが楽だが、ここではget_centerから逆算)
            start_point = current_dot.get_center()
            p_start = axes.p2c(start_point)
            x_start = p_start[1]

            # 次の点をランダムに選ぶ
            next_dot = np.random.choice(dots_by_t[j + 1])
            end_point = next_dot.get_center()
            p_end = axes.p2c(end_point)
            x_end = p_end[1]

            # ハイライト
            # scene.play(Flash(next_dot, color=YELLOW, flash_radius=0.2), run_time=0.2)

            # Brownian Bridge計算
            t_vals, x_vals = generate_brownian_bridge(
                t_start, x_start, t_end, x_end, dt=0.01, sigma=0.6
            )

            # VMobject作成
            points = [axes.c2p(t, x) for t, x in zip(t_vals, x_vals)]
            segment = VMobject()
            segment.set_points_as_corners(points)
            segment.set_color(YELLOW)
            segment.set_stroke(width=2)

            # 描画アニメーション
            scene.play(Create(segment), run_time=0.4, rate_func=linear)

            current_path_segments.add(segment)
            current_dot = next_dot

        # パス完成後、薄くして残す
        finished_paths.add(current_path_segments)
        scene.play(
            current_path_segments.animate.set_stroke(width=1, opacity=0.3).set_color(
                YELLOW
            ),
            run_time=0.2,
        )
        scene.remove(prog_text)

    scene.wait(1)


class BrownianBridgeDemo(Scene):
    def construct(self):
        run_brownian_bridge_logic(self)


class TrajectoryInference(Scene):
    def construct(self):
        run_trajectory_inference_logic(self)
