import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# --- 1. シミュレーション設定 ---
N_PARTICLES = 500  # 粒子の数（多いほど分布が綺麗になります）
T_MAX = 5.0  # シミュレーション終了時刻
DT = 0.05  # 時間刻み
TIME_STEPS = int(T_MAX / DT)

# 時間軸の作成
t_axis = np.linspace(0, T_MAX, TIME_STEPS + 1)

# SDEの数値計算（オイラー・丸山法によるブラウン運動のシミュレーション）
# positions[i, j] が、時刻 i における粒子 j の位置を表す2次元配列
positions = np.zeros((TIME_STEPS + 1, N_PARTICLES))

# ランダムな増分を計算 (正規分布 N(0, sqrt(dt)))
dW = np.sqrt(DT) * np.random.randn(TIME_STEPS, N_PARTICLES)

# 累積和をとって軌跡を計算
positions[1:, :] = np.cumsum(dW, axis=0)


# --- 2. 可視化の準備 ---
fig, (ax_paths, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("ブラウン運動: サンプルパスと周辺分布の形成", fontsize=16)

# --- 左側の設定（サンプルパス） ---
ax_paths.set_title("個々のサンプルパス (ミクロ視点)")
ax_paths.set_xlabel("時刻 t")
ax_paths.set_ylabel("位置 Xt")
ax_paths.set_xlim(0, T_MAX)
# Y軸の範囲は最終的な広がりに合わせて少し余裕を持たせる
y_limit = 3.5 * np.sqrt(T_MAX)
ax_paths.set_ylim(-y_limit, y_limit)

# 各粒子の軌跡を描画するラインの初期化（薄い色で表示）
lines = []
for _ in range(N_PARTICLES):
    (line,) = ax_paths.plot([], [], color="blue", alpha=0.1)
    lines.append(line)
# 現在時刻を示す垂直線
time_line = ax_paths.axvline(x=0, color="red", linestyle="--")


# --- 右側の設定（周辺分布ヒストグラム） ---
ax_hist.set_title("時刻 t での周辺分布 (マクロ視点)")
ax_hist.set_xlabel("位置 Xt")
ax_hist.set_ylabel("確率密度")
# ヒストグラムのX軸は、左側のY軸と合わせると分かりやすい
ax_hist.set_xlim(-y_limit, y_limit)
# Y軸（確率密度）の上限を固定（t=DTの時のピークより少し高めに設定）
ax_hist.set_ylim(0, 1.0 / np.sqrt(2 * np.pi * DT) * 1.2)

# ヒストグラムと理論曲線の初期化
patches = None
(theory_line,) = ax_hist.plot([], [], color="red", lw=2, label="理論値 (正規分布)")
ax_hist.legend()


# --- 3. アニメーション更新関数 ---
def update(frame):
    current_time = t_axis[frame]
    current_positions = positions[frame, :]

    # --- 左側の更新 ---
    # 各軌跡を現在時刻まで描画
    for i in range(N_PARTICLES):
        lines[i].set_data(t_axis[: frame + 1], positions[: frame + 1, i])
    # 現在時刻線を移動
    time_line.set_xdata([current_time, current_time])

    # --- 右側の更新 ---
    # 前のフレームのヒストグラムを削除
    global patches
    if patches is not None:
        for patch in patches:
            patch.remove()

    # 新しいヒストグラムを描画
    # density=True で確率密度関数として正規化
    n, bins, patches = ax_hist.hist(
        current_positions,
        bins=30,
        density=True,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )

    # 理論曲線（正規分布）の描画
    if current_time > 0:
        # 理論的な標準偏差 sigma = sqrt(t)
        sigma = np.sqrt(current_time)
        x_theory = np.linspace(-y_limit, y_limit, 200)
        y_theory = norm.pdf(x_theory, 0, sigma)
        theory_line.set_data(x_theory, y_theory)
        ax_hist.set_title(f"時刻 t = {current_time:.2f} での周辺分布")

    return lines + [time_line] + patches + [theory_line]


# --- 4. アニメーション実行と保存 ---
# interval はフレーム間のミリ秒数。小さいほど速い。
ani = animation.FuncAnimation(
    fig, update, frames=TIME_STEPS + 1, interval=50, blit=False
)

plt.tight_layout()
plt.show()

# 動画ファイルとして保存する場合（ffmpegが必要です）
# ani.save('brownian_motion_dist.mp4', writer='ffmpeg', fps=30, dpi=200)
# GIFとして保存する場合（ImageMagickが必要です）
# ani.save('brownian_motion_dist.gif', writer='imagemagick', fps=30)
