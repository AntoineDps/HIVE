def generate_dnn_structure(n_layers: int, mid_neurons: int, min_neurons: int):
    #! fix bug what if mid neuron is smaller than min neuron
    layers = [0] * n_layers
    min_neurons = min_neurons  #! ??? WTF

    # If n_layers is odd, there is a single middle layer, else there are two
    if n_layers % 2 == 1:
        mid_i = n_layers // 2, n_layers // 2
    else:
        mid_i = n_layers // 2 - 1, n_layers // 2

    layers[mid_i[0]] = mid_neurons
    layers[mid_i[1]] = mid_neurons

    for i in range(mid_i[0]):
        layers[mid_i[0] - (i + 1)] = int(
            layers[mid_i[0]] - (i + 1) * (mid_neurons - min_neurons) / mid_i[0]
        )
        layers[mid_i[1] + (i + 1)] = layers[mid_i[0] - (i + 1)]

    return layers
