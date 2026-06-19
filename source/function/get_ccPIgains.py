def get_ccPIgains(w, mass, ma, B, S):
    Kp = B
    Ki = (mass + ma) * w**2 - S

    return Kp, Ki
