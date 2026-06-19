import numpy as np


def fe_conv(eta, fe_irf, t_eta, t_irf=None, resample="eta"):
    # case 1: no check (dt provided)
    if t_irf is None:
        dt = t_eta
        fe = np.convolve(eta, fe_irf, mode="same") * dt

    else:
        # Convert to arrays
        t_eta = np.asarray(t_eta)
        t_irf = np.asarray(t_irf)
        eta = np.asarray(eta)
        fe_irf = np.asarray(fe_irf)

        dt_eta = round(t_eta[1] - t_eta[0], 3)
        dt_irf = round(t_irf[1] - t_irf[0], 3)

        # case 2: check dt
        if np.isclose(dt_eta, dt_irf, atol=1e-6):
            # same dt
            dt = dt_eta
            fe = np.convolve(eta, fe_irf, mode="same") * dt

        else:
            if resample == "small":
                dt = min(dt_eta, dt_irf)

            elif resample == "large":
                dt = max(dt_eta, dt_irf)

            elif resample == "eta":
                dt = dt_irf

            elif resample == "irf":
                dt = dt_eta

            else:
                raise ValueError("resample must be 'small', 'large', 'eta', or 'irf'")

            # interpolation
            new_t_eta = np.arange(t_eta[0], t_eta[-1] + dt, dt)
            new_t_irf = np.arange(t_irf[0], t_irf[-1] + dt, dt)
            eta_resampled = np.interp(new_t_eta, t_eta, eta)
            fe_irf_resampled = np.interp(new_t_irf, t_irf, fe_irf)

            fe = np.convolve(eta_resampled, fe_irf_resampled, mode="same") * dt

            # check
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(12, 6))
            # plt.subplot(2, 1, 1)
            # plt.plot(t_eta, eta, label="Original eta")
            # plt.plot(new_t_eta, eta_resampled, label="Resampled eta", linestyle="--")
            # plt.legend()
            # plt.subplot(2, 1, 2)
            # plt.plot(t_irf, fe_irf, label="Original IRF")
            # plt.plot(new_t_irf, fe_irf_resampled, label="Resampled IRF", linestyle="--")
            # plt.legend()

    return fe
