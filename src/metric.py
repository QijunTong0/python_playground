import numpy as np


def kl_divergence_zero_mean(sigma0, sigma1):
    """
    平均0の2つの多変量ガウス分布間のKLダイバージェンスを計算します。
    KL(N(0, sigma0) || N(0, sigma1))

    Parameters:
    -----------
    sigma0 : np.ndarray
        分布Pの共分散行列 (k x k)
    sigma1 : np.ndarray
        分布Qの共分散行列 (k x k)

    Returns:
    --------
    float
        KLダイバージェンスの値
    """
    k = sigma0.shape[0]

    # 1. トレース項: tr(inv(sigma1) @ sigma0)
    # np.linalg.invを使わず、solveを使うことで数値的に安定させます
    # tr(A^{-1}B) は solve(A, B) の対角和と等価です
    # sigma1 * X = sigma0 を解く
    term_trace = np.trace(np.linalg.solve(sigma1, sigma0))

    # 2. ログ行列式項: ln(|sigma1|) - ln(|sigma0|)
    # np.linalg.detの代わりにslogdetを使用してアンダーフロー/オーバーフローを防ぎます
    sign0, logdet0 = np.linalg.slogdet(sigma0)
    sign1, logdet1 = np.linalg.slogdet(sigma1)

    term_logdet = logdet1 - logdet0

    # 3. KLダイバージェンスの計算
    kl = 0.5 * (term_trace - k + term_logdet)

    return kl
