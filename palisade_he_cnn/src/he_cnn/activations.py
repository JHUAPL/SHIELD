# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import math
from copy import copy

SIN_COEFFS = [
    0,
    9.99984594193494365437e-01,
    0,
    -1.66632595072086745320e-01,
    0,
    8.31238887417884598346e-03,
    0,
    -1.93162796407356830500e-04,
    0,
    2.17326217498596729611e-06,
]
COS_COEFFS = [
    9.99971094606182687341e-01,
    0,
    -4.99837602272995734437e-01,
    0,
    4.15223086250910767516e-02,
    0,
    -1.34410769349285321733e-03,
    0,
    1.90652668840074246305e-05,
    0,
]
# you technically don't need the . to specify float division in python3
LOG_COEFFS = [
    0,
    1,
    -0.5,
    1.0 / 3,
    -1.0 / 4,
    1.0 / 5,
    -1.0 / 6,
    1.0 / 7,
    -1.0 / 8,
    1.0 / 9,
    -1.0 / 10,
]
EXP_COEFFS = [
    1,
    1,
    0.5,
    1.0 / 6,
    1.0 / 24,
    1.0 / 120,
    1.0 / 720,
    1.0 / 5040,
    1.0 / 40320,
    1.0 / 362880,
    1.0 / 3628800,
]
SIGMOID_COEFFS = [
    1.0 / 2,
    1.0 / 4,
    0,
    -1.0 / 48,
    0,
    1.0 / 480,
    0,
    -17.0 / 80640,
    0,
    31.0 / 1451520,
    0,
]


def powerOf2Extended(cipher, logDegree):
    res = [copy(cipher)]
    for i in range(logDegree):
        t = res[-1]
        res.append(t * t)
    return res


def powerExtended(cipher, degree):
    res = []
    logDegree = int(
        math.log2(degree)
    )  # both python and C++ truncate when casting float->int
    cpows = powerOf2Extended(cipher, logDegree)

    idx = 0
    for i in range(logDegree):
        powi = pow(2, i)
        res.append(cpows[i])

        for j in range(powi - 1):
            res.append(copy(res[j]))
            res[-1] *= cpows[i]

    res.append(cpows[logDegree])

    degree2 = pow(2, logDegree)

    for i in range(degree - degree2):
        res.append(copy(res[i]))
        res[-1] *= cpows[logDegree]

    return res


def polynomial_series_function(cipher, coeffs, verbose=False):
    """
    Cipher is a CKKSCiphertext, coeffs should be array-like (generally either native list or numpy array)
    """
    degree = len(coeffs)

    if verbose:
        print("initial ciphertext level = {}".format(cipher.getTowersRemaining()))

    cpows = powerExtended(cipher, degree)  # array of ciphertexts

    # cpows[0] == cipher, i.e. x^1
    res = cpows[0] * coeffs[1]  # this should be defined
    res += coeffs[0]

    for i in range(2, degree):
        coeff = coeffs[i]
        if abs(coeff) > 1e-27:
            aixi = cpows[i - 1] * coeff
            res += aixi

    if verbose:
        print("final ciphertext level = {}".format(res.getTowersRemaining()))

    return res


"""
example:

to approximate the sine function, do:
    polynomial_series_function(c1, SIN_COEFFS)
"""


def sqrt_helper(cipher, steps):
    a = copy(cipher)
    b = a - 1

    for i in range(steps):
        a *= 1 - (0.5 * b)

        # there must be a better way to do this...
        if i < steps - 1:
            b = (b * b) * (0.25 * (b - 3))

    return a


def sqrt(cipher, steps, upper_bound):
    if upper_bound == 1:
        return sqrt_helper(cipher, steps)
    return sqrt_helper(cipher * (1 / upper_bound), steps) * math.sqrt(upper_bound)


def relu(cipher, steps, upper_bound):
    x = cipher * cipher

    res = cipher + sqrt(x, steps, upper_bound)
    return 0.5 * res
