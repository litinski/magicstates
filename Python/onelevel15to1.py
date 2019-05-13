import numpy as np
from scipy import optimize
from definitions import (z, one, projx, kronecker_product, apply_rot, plog,
                         storage_x_5, storage_z_5, init5qubit, ideal15to1)


# Generates the output-state density matrix of the 15-to-1 protocol
def one_level_15to1_state(pphys, dx, dz, dm):
    # Introduce shorthand notation for logical error rate with distances dx/dz/dm
    px = plog(pphys, dx)
    pz = plog(pphys, dz)
    pm = plog(pphys, dm)

    # Step 1 of 15-to-1 protocol applying rotations 1-3 and 5
    # Last operation: apply additional storage errors due to fast faulty T measurements
    out = apply_rot(init5qubit, [one, z, one, one, one], pphys / 3, pphys / 3 + 0.5 * dz * pm, pphys / 3)
    out = apply_rot(out, [one, one, z, one, one], pphys / 3, pphys / 3 + 0.5 * dz * pm, pphys / 3)
    out = apply_rot(out, [one, one, one, z, one], pphys / 3, pphys / 3 + 0.5 * dz * pm, pphys / 3)
    out = apply_rot(out, [one, z, z, z, one], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (3 * dz) * pm, pphys / 3)
    out = storage_z_5(out, 0, (dm / dz) * pz * dm, (dm / dz) * pz * dm, (dm / dz) * pz * dm, 0)

    # Apply storage errors for dm code cycles
    out = storage_x_5(out, 0, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm, 0)
    out = storage_z_5(out, 0, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm, 0)

    # Step 2: apply rotations 6-7
    out = apply_rot(out, [z, z, z, one, one], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 2 * dz) * pm, pphys / 3)
    out = apply_rot(out, [z, z, one, z, one], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 3 * dz) * pm, pphys / 3)
    out = storage_z_5(out, (dm / dx) * px * dm, (dm / dz) * pz * dm, 0.5 * (dm / dz) * pz * dm,
                      0.5 * (dm / dz) * pz * dm, 0)

    # Apply storage errors for dm code cycles
    out = storage_x_5(out, 0.5 * px * dm, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm,
                      0.5 * (dz / dx) * px * dm, 0)
    out = storage_z_5(out, 0.5 * px * dm, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm,
                      0.5 * (dx / dz) * pz * dm, 0)

    # Step 3: apply rotation 4 and 8-9
    out = apply_rot(out, [z, one, z, z, one], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 3 * dz) * pm, pphys / 3)
    out = apply_rot(out, [z, one, one, z, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 4 * dz) * pm, pphys / 3)
    out = apply_rot(out, [one, one, one, one, z], pphys / 3, pphys / 3 + 0.5 * dz * pm, pphys / 3)
    out = storage_z_5(out, (dm / dx) * px * dm, 0, 0.5 * (dm / dz) * pz * dm, (dm / dz) * pz * dm,
                      0.5 * (dm / dz) * pz * dm)

    # Apply storage errors for dm code cycles
    out = storage_x_5(out, 0.5 * px * dm, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm,
                      0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm)
    out = storage_z_5(out, 0.5 * px * dm, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm,
                      0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm)

    # Step 4: apply rotations 10-11
    out = apply_rot(out, [z, z, one, one, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 4 * dz) * pm, pphys / 3)
    out = apply_rot(out, [z, one, z, one, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 4 * dz) * pm, pphys / 3)
    out = storage_z_5(out, (dm / dx) * px * dm, 0.5 * (dm / dz) * pz * dm, 0.5 * (dm / dz) * pz * dm, 0,
                      (dm / dz) * pz * dm)

    # Apply storage errors for dm code cycles
    out = storage_x_5(out, 0.5 * px * dm, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm,
                      0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm)
    out = storage_z_5(out, 0.5 * px * dm, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm,
                      0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm)

    # Step 5: apply rotations 12-13
    out = apply_rot(out, [z, z, z, z, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (dx + 4 * dz) * pm, pphys / 3)
    out = apply_rot(out, [one, one, z, z, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (3 * dz) * pm, pphys / 3)
    out = storage_z_5(out, 0.5 * (dm / dx) * px * dm, 0.5 * (dm / dz) * pz * dm, (dm / dz) * pz * dm,
                      (dm / dz) * pz * dm, (dm / dz) * pz * dm)

    # Apply storage errors for dm code cycles
    # Qubit 1 is consumed as an output state: additional storage errors for dx code cycles
    out = storage_x_5(out, 0.5 * px * (dx + 2 * dm), 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm,
                      0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm)
    out = storage_z_5(out, 0.5 * px * (dx + 2 * dm), 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm,
                      0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm)

    # Step 6: apply rotation 14-15
    out = apply_rot(out, [one, z, one, z, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (4 * dz) * pm, pphys / 3)
    out = apply_rot(out, [one, z, z, one, z], pphys / 3 + 0.5 * pm * dm, pphys / 3 + (4 * dz) * pm, pphys / 3)
    out = storage_z_5(out, 0, (dm / dz) * pz * dm, 0.5 * (dm / dz) * pz * dm, 0.5 * (dm / dz) * pz * dm,
                      (dm / dz) * pz * dm)

    # Apply storage errors for dm code cycles
    out = storage_x_5(out, 0, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm, 0.5 * (dz / dx) * px * dm,
                      0.5 * (dz / dx) * px * dm)
    out = storage_z_5(out, 0, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm, 0.5 * (dx / dz) * pz * dm,
                      0.5 * (dx / dz) * pz * dm)

    return out


# Calculates the output error and cost of the 15-to-1 protocol with a physical error rate pphys
# and distances dx, dz and dm
def cost_of_one_level_15to1(pphys, dx, dz, dm):
    # Generate output state of 15-to-1 protocol
    out = one_level_15to1_state(pphys, dx, dz, dm)

    # Compute failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail = np.real(1 - np.trace(np.dot(kronecker_product([one, projx, projx, projx, projx]), out)))

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel = 1 / (1 - pfail) * np.dot(np.dot(kronecker_product([one, projx, projx, projx, projx]), out),
                                          kronecker_product([one, projx, projx, projx, projx]).conj().transpose())

    # Compute output error from the infidelity between the post-selected state and the ideal output state
    pout = np.real(1 - np.trace(np.dot(outpostsel, ideal15to1)))

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
        return 231 / pout * d * plog(pphys, d) - 0.01

    def logerr2(d):
        return 20284 / pout * d * plog(pphys, d) - 0.01

    reqdist1 = int(2 * round(optimize.root(logerr1, 3, method='hybr').x[0] / 2) + 1)
    reqdist2 = int(2 * round(optimize.root(logerr2, 3, method='hybr').x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    print('15-to-1 with pphys=', pphys, ', dx=', dx, ', dz=', dz, ', dm=', dm, sep='')
    print('Output error: ', '%.4g' % pout, sep='')
    print('Failure probability: ', '%.3g' % pfail, sep='')
    print('Qubits: ', 2 * (dx + 4 * dm) * (dx + 4 * dz), sep='')
    print('Code cycles: ', '%.2f' % (6 * dm / (1 - pfail)), sep='')
    print('Space-time cost: ', '%.0f' % (2 * (4 * dm + dx) * (dx + 4 * dz) * 6 * dm / (1 - pfail)), ' qubitcycles',
          sep='')
    print('For a 100-qubit computation: ',
          ('%.3f' % ((4 * dm + dx) * (dx + 4 * dz) * 6 * dm / (1 - pfail) / reqdist1 ** 3)), 'd^3 (d=', reqdist1, ')',
          sep='')
    print('For a 5000-qubit computation: ',
          ('%.3f' % ((4 * dm + dx) * (dx + 4 * dz) * 6 * dm / (1 - pfail) / reqdist2 ** 3)), 'd^3 (d=', reqdist2, ')',
          sep='')
    print('')
