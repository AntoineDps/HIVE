import numpy as np
from scipy import signal
import pandas as pd


def eta2all(t, eta, freq="f", wave_type="irr", S_type="JONSWAP", gamma=None):
    x, S_fft, S_fft_filt, S_welch = eta2S(t, eta, freq=freq)
    Te, Hs = S2param(x, S_fft)
    J = Eflux(Te, Hs, wave_type)
    eps = steepness(Te, Hs)
    S_th = param2S(Te, Hs, x, gamma=gamma, type=S_type)

    if freq == "w":
        x_type = "w"
    elif freq == "f":
        x_type = "f"

    wave_params = pd.DataFrame(
        {
            x_type: x,
            "S_fft": S_fft,
            "S_fft_filt": S_fft_filt,
            "S_welch": S_welch,
            "S_th": S_th,
            "Te": Te,
            "Hs": Hs,
            "J": J,
            "eps": eps,
            "S_type": S_type,
            "gamma": gamma,
        }
    )

    return wave_params


def eta2S(t, eta, freq="f"):

    # TODO : test // f or w // welch implement // welch, fft, fft_filt compare

    # Sampling
    dt = t[1] - t[0]
    fs = 1.0 / dt
    N = len(eta)

    # FFT
    z = np.abs(np.fft.fft(eta))
    Nz = len(z)
    df = fs / N

    # remove negatives and DC component
    z = z[Nz // 2 + 1 :]
    Nz = len(z)

    # multipply by 2 and account for N
    A = z * 2 / N

    # spectral frequencies
    S_f_fft = 0.5 * A**2 / df

    # frequency vector
    f_fft = np.arange(df, df * (Nz + 1), df)

    # FFT filtered

    # moving average filter
    window_size = 5
    S_f_fft_filt = signal.convolve(
        S_f_fft, np.ones(window_size) / window_size, mode="same"
    )
    # WELCH
    nperseg = 4024

    f_welch, S_f_welch = signal.welch(
        eta.to_numpy(), fs=fs, window="hann", nperseg=nperseg, detrend="constant"
    )

    # remove DC component
    f_welch = f_welch[1:]
    S_f_welch = S_f_welch[1:]
    f_welch = f_fft  #!
    S_f_welch = S_f_fft  #!

    # freq variable
    if freq == "w":
        w_fft = 2.0 * np.pi * f_fft
        S_w_fft = S_f_fft / (2.0 * np.pi)
        S_w_fft_filt = S_f_fft_filt / (2.0 * np.pi)
        S_w_welch = S_f_welch / (2.0 * np.pi)
        return w_fft, S_w_fft, S_w_fft_filt, S_w_welch
    else:
        return f_fft, S_f_fft, S_f_fft_filt, S_f_welch


def S2param(x, S):

    df = x[1] - x[0]
    m0 = np.sum(S * df)
    m_minus1 = np.sum((x**-1) * S * df)

    Hs = round(4 * np.sqrt(m0), 2)
    Te = round(m_minus1 / m0, 2)

    return Te, Hs


def param2S(T, H, x, freq="f", gamma=3.3, type="JONSWAP"):  #! w or f
    """
    Convert wave parameters to power spectral density S(w) or S(f)
    """

    if type == "JONSWAP":
        if freq == "w":
            w = x
            wp = 2 * np.pi / (T / 0.903)
            alpha = 1 - 0.287 * np.log(gamma)
            theta = np.zeros_like(w)
            theta[w <= wp] = 0.07
            theta[w > wp] = 0.09

            S = (
                alpha
                * 5
                / 16
                * wp**4
                / w**5
                * H**2
                * np.exp(-5 / 4 * (wp / w) ** 4)
                * gamma ** np.exp(-((w - wp) ** 2) / (2 * theta**2 * wp**2))
            )

        elif freq == "f":
            f = x
            fp = 1 / (T / 0.903)
            alpha = 1 - 0.287 * np.log(gamma)
            theta = np.zeros_like(f)
            theta[f <= fp] = 0.07
            theta[f > fp] = 0.09

            S = (
                alpha
                * 5
                / 16
                * fp**4
                / f**5
                * H**2
                * np.exp(-5 / 4 * (fp / f) ** 4)
                * gamma ** np.exp(-((f - fp) ** 2) / (2 * theta**2 * fp**2))
            )

    return S


def S2eta(x, S, t, freq="f"):

    L = len(S)
    phi = (np.random.rand(L) - 0.5) * 2 * np.pi  # random phases in [-π, π]
    eta = np.zeros_like(t)
    A = []

    if freq == "w":
        w = x
        dw = w[1] - w[0]

        for i in range(L):  # Sum all harmonic components
            a_i = np.sqrt(2 * S[i] * dw)
            w_i = w[i]
            eta += a_i * np.cos(w_i * t - phi[i])
            A.append(a_i)

    elif freq == "f":
        f = x
        df = f[1] - f[0]

        for i in range(L):  # Sum all harmonic components
            a_i = np.sqrt(2 * S[i] * df)
            f_i = f[i]
            eta += a_i * np.cos(2 * np.pi * f_i * t - phi[i])
            A.append(a_i)

    return t, eta, phi, A


def steepness(T, H):
    g = 9.81

    eps = H / (g * T**2)

    return eps


def Eflux(T, H, type, rho=1025):
    rho = rho
    g = 9.81

    if type == "reg":
        # Energy density [J/m²]
        J = rho * g**2 * H**2 * T / (32 * np.pi)
    elif type == "irr":
        # Energy density [J/m²]
        J = rho * g**2 * H**2 * T / (64 * np.pi)

    return J


def CWR(P, J, r):
    CWR = P / (J * 2 * r)
    return CWR


def WaveJ(T, H, type, rho=1025):
    """
    Compute wave energy flux J [W/m] for given wave parameters.

    Parameters
    ----------
    T : float
        Wave period [s].
    H : float
        Wave height [m].
    type : str
        Wave type: 'reg' for regular waves, 'irr' for irregular waves.
    rho : float, optional
        Water density [kg/m³]. Default is 1025 kg/m³ (seawater).

    Returns
    -------
    J : float
        Wave energy flux [W/m].
    """
    g = 9.81  # Acceleration due to gravity [m/s²]

    if type == "reg":
        # Energy flux for regular waves
        J = rho * g**2 * H**2 * T / (32 * np.pi)
    elif type == "irr":
        # Energy flux for irregular waves
        J = rho * g**2 * H**2 * T / (64 * np.pi)
    else:
        raise ValueError("Invalid wave type. Use 'reg' or 'irr'.")

    return J
