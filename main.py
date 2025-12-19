import numpy as np
from manim import (
    BLUE,
    DOWN,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    WHITE,
    Axes,
    Create,
    FadeIn,
    Line,
    Scene,
    Text,
    VGroup,
)


class SDESimulation(Scene):
    def construct(self):
        # --- 1. シミュレーション設定 ---
        dt = 0.5  # 時間刻み（誤差を目立たせるため大きめに設定）
        total_time = 4.0
        steps = int(total_time / dt)

        # 座標系の設定
        ax = Axes(
            x_range=[0, total_time + 0.5, 1],  # 少し余裕を持たせる
            y_range=[-3, 8, 2],  # 範囲を少し調整
            axis_config={"include_tip": True},
            x_length=10,
            y_length=6,
        ).add_coordinates()

        labels = ax.get_axis_labels(x_label="t", y_label="X_t")
        title = Text("SDE Simulation: Euler-Maruyama vs Exact", font_size=32).to_edge(
            UP
        )
        self.add(ax, labels, title)

        # 初期値 Z_0 = [0, 1] (X_0=0, Y_0=1)
        x_0, y_0 = 0.0, 1.0

        # --- 2. データ生成 ---

        # A. オイラー・丸山法 (Euler-Maruyama)
        # 近似: X_{t+dt} = X_t + Y_t * dt
        em_points = [(0, x_0)]
        curr_x, curr_y = x_0, y_0

        # 比較のためシードを固定（このシードでの軌道が「近似解」）
        np.random.seed(42)
        for _ in range(steps):
            dw = np.random.normal(0, np.sqrt(dt))
            curr_x = curr_x + curr_y * dt
            curr_y = curr_y + dw
            em_points.append((em_points[-1][0] + dt, curr_x))

        # B. 厳密シミュレーション (Exact Simulation)
        # 理論解: Z_{t+dt} ~ N(Mean, Cov)
        exact_points = [(0, x_0)]
        curr_x, curr_y = x_0, y_0

        # 注: 単純に同じシードを使っても、乱数の消費の仕方が違うため(1個 vs 2個)、
        # 同じブラウン運動パスにはなりません。ここでは「統計的な振る舞いの違い」を示します。
        np.random.seed(10)  # 別のシードで見やすい軌道を選択
        for _ in range(steps):
            # 共分散行列 Q_dt
            # [[ t^3/3, t^2/2 ],
            #  [ t^2/2, t     ]]
            cov = np.array([[(1 / 3) * dt**3, (1 / 2) * dt**2], [(1 / 2) * dt**2, dt]])
            mean = np.array([curr_x + curr_y * dt, curr_y])

            # 多変量正規分布からサンプリング
            next_val = np.random.multivariate_normal(mean, cov)
            curr_x, curr_y = next_val[0], next_val[1]
            exact_points.append((exact_points[-1][0] + dt, curr_x))

        # --- 3. アニメーション要素の作成 ---

        # オイラー法のパス（オレンジ）
        em_path = VGroup()
        for i in range(len(em_points) - 1):
            line = Line(
                ax.c2p(em_points[i][0], em_points[i][1]),
                ax.c2p(em_points[i + 1][0], em_points[i + 1][1]),
                color=ORANGE,
                stroke_width=4,
            )
            em_path.add(line)

        # 厳密解のパス（青）
        exact_path = VGroup()
        for i in range(len(exact_points) - 1):
            line = Line(
                ax.c2p(exact_points[i][0], exact_points[i][1]),
                ax.c2p(exact_points[i + 1][0], exact_points[i + 1][1]),
                color=BLUE,
                stroke_width=4,
            )
            exact_path.add(line)

        em_label = (
            Text("Euler-Maruyama (Approximation)", color=ORANGE, font_size=24)
            .to_corner(RIGHT + UP)
            .shift(DOWN * 1.5)
        )
        exact_label = Text(
            "Exact Simulation (Covariance)", color=BLUE, font_size=24
        ).next_to(em_label, DOWN, align_edge=LEFT)

        # --- 4. 描画実行 ---

        # オイラー法の描画
        self.play(FadeIn(em_label))
        # 順次描画することで時間発展を表現
        for line in em_path:
            self.play(Create(line), run_time=0.3, rate_func=lambda t: t)  # 線形な速度で

        self.wait(0.5)

        # 厳密解の描画
        self.play(FadeIn(exact_label))
        for line in exact_path:
            self.play(Create(line), run_time=0.3, rate_func=lambda t: t)

        # 解説テキスト
        info = Text(
            "Euler method assumes constant velocity during dt,\ncausing integration errors.",
            font_size=24,
            color=WHITE,
        ).to_edge(DOWN)

        self.play(FadeIn(info))
        self.wait(3)
