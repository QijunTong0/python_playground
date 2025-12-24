import matplotlib.pyplot as plt
import numpy as np
import ot
from matplotlib import gridspec


def compute_entropic_ot_coupling(epsilon=1.0, n_bins=100):
    """
    2つの1次元混合ガウス分布を作成し、エントロピー正則化最適輸送行列(カップリング)を計算する関数

    Returns:
        grid (np.array): 離散化された座標 [-4, 4]
        pi (np.array): 最適輸送カップリング行列 (n_bins x n_bins)
        a (np.array): ソース側の分布 (ヒストグラム)
        b (np.array): ターゲット側の分布 (ヒストグラム)
    """
    # 1. グリッドの作成 [-4, 4]を100等分
    grid = np.linspace(-4, 4, n_bins)

    # 2. 混合ガウス分布の定義 (確率密度関数)
    def gmm_pdf(x, means, covariances, weights):
        pdf = np.zeros_like(x)
        for m, s, w in zip(means, covariances, weights):
            # ガウス分布の計算
            pdf += w * (1 / np.sqrt(2 * np.pi * s)) * np.exp(-((x - m) ** 2) / (2 * s))
        return pdf

    # 分布a (Source): 左側に偏った2つの山
    a = gmm_pdf(grid, means=[-2.5, -0.5], covariances=[0.2, 1.1], weights=[0.6, 0.4])

    # 分布b (Target): 右側に移動した2つの山
    b = gmm_pdf(grid, means=[1.0, 2.5], covariances=[1.1, 0.3], weights=[0.5, 0.5])

    # 正規化 (離散分布として和を1にする)
    a = a / np.sum(a)
    b = b / np.sum(b)

    # 3. コスト行列の作成 (二乗ユークリッド距離)
    # ot.distはデフォルトで二乗ユークリッド距離を計算します
    # shapeを(n, 1)に変換して渡す必要があります
    M = ot.dist(grid.reshape(-1, 1), grid.reshape(-1, 1), metric="sqeuclidean")

    # 4. エントロピー正則化最適輸送 (Sinkhornアルゴリズム)
    # reg が epsilon に相当します
    pi = ot.sinkhorn(a, b, M, reg=epsilon)

    return grid, pi, a, b


def sample_from_coupling(pi, grid, n_samples):
    """
    カップリング行列 pi から (x, y) のペアをサンプリングする関数

    Args:
        pi (np.array): カップリング行列 (同時確率分布 P(x, y))
        grid (np.array): グリッドの座標値
        n_samples (int): サンプリングする個数

    Returns:
        samples (np.array): サンプリングされた座標ペア (n_samples, 2)
    """
    # カップリング行列を1次元配列（ベクトル）に平坦化
    pi_flat = pi.flatten()

    # 確率の合計が数値誤差で厳密に1にならない場合があるため、正規化して修正
    pi_flat = pi_flat / np.sum(pi_flat)

    # 全要素数分のインデックス (0 ~ 9999)
    indices = np.arange(pi.size)

    # 1. 確率 pi_flat に基づいてインデックスをサンプリング
    sampled_indices = np.random.choice(indices, size=n_samples, p=pi_flat)

    # 2. 平坦化されたインデックスを元の (row, col) インデックスに戻す
    # row_idx が x (Source), col_idx が y (Target) に対応
    row_idx, col_idx = np.unravel_index(sampled_indices, pi.shape)

    # 3. インデックスを実際のグリッド座標に変換
    x_samples = grid[row_idx]
    y_samples = grid[col_idx]

    # 結果を結合して (N, 2) の形にする
    samples = np.column_stack((x_samples, y_samples))

    return samples


def plot_coupling_with_marginals(grid, pi, a, b, title="Entropic OT Coupling"):
    """
    カップリング行列を中心に、上にターゲット分布(b)、左にソース分布(a)を表示する関数
    """

    # 図の定義 (サイズは適宜調整)
    fig = plt.figure(figsize=(10, 10))

    # グリッドレイアウトの定義 (2行2列)
    # width_ratios, height_ratios で周辺分布とメイン行列のサイズ比率を調整
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1, 4], height_ratios=[1, 4], wspace=0.05, hspace=0.05
    )  # グラフ間の隙間を小さくする

    # 1. 左上の領域 (通常は空けるか、タイトルなどを入れる)
    ax_null = plt.subplot(gs[0, 0])
    ax_null.axis("off")
    ax_null.text(
        0.5, 0.5, "Source (y)\nvs\nTarget (x)", ha="center", va="center", fontsize=10
    )

    # 2. 上の領域: ターゲット分布 b (横軸 x)
    ax_top = plt.subplot(gs[0, 1])
    ax_top.plot(grid, b, color="blue", label="Target b")
    ax_top.fill_between(grid, 0, b, color="blue", alpha=0.3)
    ax_top.set_xlim(-4, 4)
    ax_top.set_xticks([])  # 目盛りを消す（下のグラフと共有するため）
    ax_top.set_yticks([])  # y軸目盛りも消してスッキリさせる
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    # ax_top.set_title("Target Distribution (b)", fontsize=12)

    # 3. 左の領域: ソース分布 a (縦軸 y -> 90度回転)
    ax_left = plt.subplot(gs[1, 0])
    # xとyを入れ替えてプロットすることで90度回転を実現
    ax_left.plot(a, grid, color="red", label="Source a")
    ax_left.fill_betweenx(grid, 0, a, color="red", alpha=0.3)  # 横方向に塗りつぶし
    ax_left.set_ylim(-4, 4)
    ax_left.invert_xaxis()  # 分布の「山」を中心（右側）に向けるための反転
    ax_left.set_xticks([])  # 目盛りを消す
    # ax_left.set_ylabel("Source Distribution (a)", fontsize=12)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["left"].set_visible(False)
    ax_left.spines["bottom"].set_visible(False)

    # 4. 中央の領域: カップリング行列 pi
    ax_main = plt.subplot(gs[1, 1])
    # extentで座標軸をグリッドに合わせる
    im = ax_main.imshow(
        pi,
        interpolation="nearest",
        origin="lower",
        extent=[-4, 4, -4, 4],
        cmap="Purples",
    )

    ax_main.set_xlabel("Target Domain (x)")
    ax_main.set_ylabel("Source Domain (y)")
    ax_main.yaxis.set_label_position("right")  # Y軸ラベルを右側に
    ax_main.yaxis.tick_right()  # Y軸目盛りを右側に

    # タイトルなどを設定
    plt.suptitle(title, y=0.92, fontsize=16)
    plt.show()
