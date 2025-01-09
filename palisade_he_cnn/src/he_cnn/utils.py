# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import shutil
from pathlib import Path

import numpy as np
import pyOpenFHE as pal
from pyOpenFHE import CKKS as pal_ckks
import numpy as np
from pathlib import Path

serial = pal_ckks.serial


def is_power_of_2(x):
    return x > 0 and x & (x - 1) == 0


def next_power_of_2(n):
    p = 1
    if n and not (n & (n - 1)):
        return n
    while p < n:
        p <<= 1
    return p


def load_cc_and_keys(batch_size, mult_depth=10, scale_factor_bits=40, bootstrapping=False):
    f = "{}-{}-{}-{}".format(batch_size, mult_depth, scale_factor_bits, int(bootstrapping))
    path = Path("serialized") / f

    P = (path / "PublicKey.bin").as_posix()
    publicKey = pal_ckks.serial.DeserializeFromFile_PublicKey(P, pal_ckks.serial.SerType.BINARY)
    P = (path / "PrivateKey.bin").as_posix()
    secretKey = pal_ckks.serial.DeserializeFromFile_PrivateKey(P, pal_ckks.serial.SerType.BINARY)

    keys = pal_ckks.KeyPair(publicKey, secretKey)
    cc = publicKey.getCryptoContext()

    P = (path / "EvalMultKey.bin").as_posix()
    pal_ckks.serial.DeserializeFromFile_EvalMultKey_CryptoContext(cc, P, pal_ckks.serial.SerType.BINARY)
    P = (path / "EvalAutomorphismKey.bin").as_posix()
    pal_ckks.serial.DeserializeFromFile_EvalAutomorphismKey_CryptoContext(cc, P, pal_ckks.serial.SerType.BINARY)

    if bootstrapping:
        cc.evalBootstrapSetup()

    return cc, keys


def save_cc_and_keys(cc, keys, path):
    P = (path / "PublicKey.bin").as_posix()
    assert pal_ckks.serial.SerializeToFile(P, keys.publicKey, pal_ckks.serial.SerType.BINARY)
    P = (path / "PrivateKey.bin").as_posix()
    assert pal_ckks.serial.SerializeToFile(P, keys.secretKey, pal_ckks.serial.SerType.BINARY)
    P = (path / "EvalMultKey.bin").as_posix()
    assert pal_ckks.serial.SerializeToFile_EvalMultKey_CryptoContext(cc, P, pal_ckks.serial.SerType.BINARY)
    P = (path / "EvalAutomorphismKey.bin").as_posix()
    assert pal_ckks.serial.SerializeToFile_EvalAutomorphismKey_CryptoContext(cc, P, pal_ckks.serial.SerType.BINARY)


def create_cc_and_keys(batch_size, mult_depth=10, scale_factor_bits=40, bootstrapping=False, save=False):
    # We make use of palisade HE by creating a crypto context object
    # this specifies things like multiplicative depth
    cc = pal_ckks.genCryptoContextCKKS(
        mult_depth,  # number of multiplications you can perform
        scale_factor_bits,  # kindof like number of bits of precision
        batch_size,  # length of your vector, can be any power-of-2 up to 2^14
    )

    print(f"CKKS scheme is using ring dimension = {cc.getRingDimension()}, batch size = {cc.getBatchSize()}")

    cc.enable(pal.enums.PKESchemeFeature.PKE)
    cc.enable(pal.enums.PKESchemeFeature.KEYSWITCH)
    cc.enable(pal.enums.PKESchemeFeature.LEVELEDSHE)
    cc.enable(pal.enums.PKESchemeFeature.ADVANCEDSHE)
    cc.enable(pal.enums.PKESchemeFeature.FHE)

    # generate keys
    keys = cc.keyGen()
    cc.evalMultKeyGen(keys.secretKey)
    cc.evalPowerOf2RotationKeyGen(keys.secretKey)

    if bootstrapping:
        cc.evalBootstrapSetup()
        cc.evalBootstrapKeyGen(keys.secretKey)

    if save:
        f = "{}-{}-{}-{}".format(batch_size, mult_depth, scale_factor_bits, int(bootstrapping))
        path = Path("serialized") / f
        path.mkdir(parents=True, exist_ok=True)
        save_cc_and_keys(cc, keys, path)

    return cc, keys


def get_keys(mult_depth,
             scale_factor_bits,
             batch_size,
             bootstrapping):
    try:
        cc, keys = load_cc_and_keys(batch_size,
                                    mult_depth=mult_depth,
                                    scale_factor_bits=scale_factor_bits,
                                    bootstrapping=bootstrapping)
    except:
        cc, keys = create_cc_and_keys(batch_size,
                                      mult_depth=mult_depth,
                                      scale_factor_bits=scale_factor_bits,
                                      bootstrapping=bootstrapping,
                                      save=True)
    return cc, keys


def get_filters_and_biases_from_conv2d(layer):
    filters = layer.weight.detach().numpy()
    if hasattr(layer, "bias") and layer.bias is not None:
        biases = layer.bias.detach().numpy()
    else:
        # without bias
        # same as number of output channels (each bias is broadcast over the channel)
        biases = np.zeros((filters.shape[0],))

    filters = filters.transpose(1, 0, 2, 3)
    pad_to = next_power_of_2(filters.shape[0])

    if pad_to is not None:
        if filters.shape[0] < pad_to:
            filters = np.concatenate(
                [filters, np.zeros((pad_to - filters.shape[0],) + filters.shape[1:])]
            )

    return filters, biases


def get_scale_and_shift_from_bn(layer):
    mu = layer.running_mean.detach().numpy()
    var = layer.running_var.detach().numpy()
    gamma = (
        layer.weight.detach().numpy()
    )  # https://discuss.pytorch.org/t/getting-parameters-of-torch-nn-batchnorm2d-during-training/38913/3
    beta = layer.bias.detach().numpy()
    eps = layer.eps

    sigma = np.sqrt(var + eps)  # std dev

    # compute scale factor
    scale = gamma / sigma

    # compute shift factor
    shift = -gamma * mu / sigma + beta

    return scale, shift


# needs to know either number of channels or matrix size
def get_weights_and_biases_from_linear(layer, mtx_size, bias, pool_factor=1):
    nout = layer.weight.size(0)
    weights = layer.weight.detach().numpy()
    num_channels = weights.shape[1] // (mtx_size * mtx_size)
    weights = weights.reshape(nout, num_channels, mtx_size, mtx_size)
    weights = weights.reshape(nout, -1)
    weights = np.repeat(weights, pool_factor * pool_factor, axis=1)

    if bias:
        biases = layer.bias.detach().numpy()
    else:
        biases = np.zeros(nout)

    return weights, biases


# Given a model and an input, get intermediate layer output
def get_intermediate_output(model, layer, inputs):
    layer_name = "layer"
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    layer.register_forward_hook(
        get_activation(layer_name)
    )
    _ = model(inputs)
    return activation[layer_name]


def compare_accuracy(keys, cnn_context, unencrypted, name="block", num_digits=4):
    A = decrypt_and_reshape(cnn_context, keys.secretKey, cnn_context.mtx_size)
    B = unencrypted.detach().cpu().numpy()[0]
    diff = np.abs(A - B[cnn_context.permutation])
    print(f"error in {name}:\nmax  = {np.max(diff):.0{num_digits}f}\nmean = {np.mean(diff):.0{num_digits}f}")


def decrypt_and_reshape(cnn_context, secret_key, mtx_size):
    cc = secret_key.getCryptoContext()
    decrypted_output = [cc.decrypt(secret_key, ctxt) for ctxt in cnn_context.shards]
    decrypted_output = np.hstack(decrypted_output)
    num_out_chan = int(round(len(decrypted_output) / (mtx_size * mtx_size)))
    decrypted_output = decrypted_output.reshape((num_out_chan, mtx_size, mtx_size))
    decrypted_output = decrypted_output[0:: cnn_context.duplication_factor]

    return decrypted_output


def serialize(cc, keys, ctxt):
    path = Path("serialized")
    path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    assert serial.SerializeToFile("serialized/CryptoContext.bin", ctxt, serial.SerType.BINARY)
    assert serial.SerializeToFile("serialized/ciphertext.bin", ctxt, serial.SerType.BINARY)
    assert serial.SerializeToFile("serialized/PublicKey.bin", keys.publicKey, serial.SerType.BINARY)
    assert serial.SerializeToFile("serialized/PrivateKey.bin", keys.secretKey, serial.SerType.BINARY)
    assert serial.SerializeToFile_EvalMultKey_CryptoContext(cc, "serialized/EvalMultKey.bin", serial.SerType.BINARY)
    assert serial.SerializeToFile_EvalAutomorphismKey_CryptoContext(cc, "serialized/EvalAutomorphismKey.bin",
                                                                    serial.SerType.BINARY)


if __name__ == "__main__":
    cc, keys = get_keys(mult_depth=34, scale_factor_bits=59, batch_size=32 * 32 * 32, bootstrapping=True)
    print(cc.getBatchSize())
    shard = cc.encrypt(keys.publicKey, [0.0 for _ in range(32768)])
    serialize(cc, keys, shard)
