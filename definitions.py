import numpy as np

# Pauli matrices and projector |+><+|
x = np.array([[0, 1], [1, 0]])
y = np.array([[0, -1j], [1j, 0]])
z = np.array([[1, 0], [0, -1]])
one = np.array([[1, 0], [0, 1]])
projx = (one + x) / 2

# Density matrices of pure |+> states, pure magic states and pure CCZ states
plusstate = np.dot(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]), np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)]]))
magicstate = np.dot(np.array([[1 / np.sqrt(2)], [np.exp(1j * np.pi / 4) * 1 / np.sqrt(2)]]),
                    np.array([[1 / np.sqrt(2), np.exp(-1j * np.pi / 4) * 1 / np.sqrt(2)]]))
CCZstate = np.dot(np.array([[1 / np.sqrt(8)], [1 / np.sqrt(8)], [1 / np.sqrt(8)], [1 / np.sqrt(8)], [1 / np.sqrt(8)],
                            [1 / np.sqrt(8)], [1 / np.sqrt(8)], [-1 / np.sqrt(8)]]), np.array([[1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                1 / np.sqrt(8),
                                                                                                -1 / np.sqrt(8)]]))


# Computes the tensor product of a list of matrices
def kronecker_product(matrices):
    res = np.kron(matrices[0], matrices[1])
    for i in matrices[2:]:
        res = np.kron(res, i)
    return res


# Density matrices of 5, 7 and 4 |+> states
init5qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate, plusstate])
init7qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate, plusstate, plusstate, plusstate])
init4qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate])

# Density matrices corresponding to the ideal output state of 15-to-1, 20-to-4 and 8-to-CCZ
ideal15to1 = kronecker_product([magicstate, plusstate, plusstate, plusstate, plusstate])
ideal20to4 = kronecker_product([magicstate, magicstate, magicstate, magicstate, plusstate, plusstate, plusstate])
ideal8toCCZ = kronecker_product([CCZstate, plusstate])


# Pauli product rotation e^(iP*phi), where the Pauli product P is specified by 'axis' and phi is the rotation angle
def pauli_rot(axis, angle):
    return np.cos(angle) * np.eye(2 ** len(axis)) + 1j * np.sin(angle) * kronecker_product(axis)


# Applies a pi/8 Pauli product rotation specified by 'axis' with probability 1-p1-p2-p3
# A P_(pi/2) / P_(-pi/4) / P_(pi/4) error occurs with probability p1 / p2 / p3
def apply_rot(state, axis, p1, p2, p3):
    return (1 - p1 - p2 - p3) * np.dot(np.dot(pauli_rot(axis, np.pi / 8), state),
                                       pauli_rot(axis, np.pi / 8).conj().transpose()) \
           + p1 * np.dot(np.dot(pauli_rot(axis, 5 * np.pi / 8), state),
                         pauli_rot(axis, 5 * np.pi / 8).conj().transpose()) \
           + p2 * np.dot(np.dot(pauli_rot(axis, -1 * np.pi / 8), state),
                         pauli_rot(axis, -1 * np.pi / 8).conj().transpose()) \
           + p3 * np.dot(np.dot(pauli_rot(axis, 3 * np.pi / 8), state),
                         pauli_rot(axis, 3 * np.pi / 8).conj().transpose())


# Applies a Pauli operator to a state with probability p
def apply_pauli(state, pauli, p):
    return (1 - p) * state + p * np.dot(np.dot(kronecker_product(pauli), state), kronecker_product(pauli))


# Estimate of the logical error rate of a surface-code patch with code distance d and circuit-level error rate pphys
def plog(pphys, d):
    return 0.1 * (100 * pphys) ** ((d + 1) / 2)


# For the 8-to-CCZ protocol, applies X/Z storage errors to qubits 1-4 with probabilities p1-p4
def storage_x_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [x, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one], p2)
    res = apply_pauli(res, [one, one, x, one], p3)
    res = apply_pauli(res, [one, one, one, x], p4)
    return res


def storage_z_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [z, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one], p2)
    res = apply_pauli(res, [one, one, z, one], p3)
    res = apply_pauli(res, [one, one, one, z], p4)
    return res


# For the 15-to-1 protocol, applies X/Z storage errors to qubits 1-5 with probabilities p1-p5
def storage_x_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [x, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one], p4)
    res = apply_pauli(res, [one, one, one, one, x], p5)
    return res


def storage_z_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [z, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one], p4)
    res = apply_pauli(res, [one, one, one, one, z], p5)
    return res


# For the 20-to-4 protocol, applies X/Z storage errors to qubits 1-7 with probabilities p1-p7
def storage_x_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [x, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, x, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, x, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, x], p7)
    return res


def storage_z_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [z, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, z, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, z, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, z], p7)
    return res
