import scipy.sparse.linalg

from RadicalA import *
from RadicalB import *

# NB; The order of spins is {Nb1 Nb2 Nb3... EA Na1 Na2 Na3... } { Nb1 Nb2 Nb3... EA Na1 Na2 Na3...} - i.e. {RadicalA}{RadicalB}
# This allows for simplification of our spin systems to consider only a single radical (i.e. a FAD-Z type system)

# Further, we can subdivide a single radical into two subsystem {a} and {b} and calculate eigenvalues of a single
# subsystem (involving only {Nb1 Nb2 Nb3... EA} or only {EA Na1 Na2 Na3...} and if the Ha and Hb ~ commute (which occurs
# when the HFIT's involved are naturally polarised strongly along some axis, then the eigenvalues of the total system
# can be approximated as a sum of eigenvalues of the two subsystems - this follows from the Spectral Theorem, and in
# particular Weyl's inequality tells us that this process always overestimates Vmax

N5 = RadicalA('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
N10 = RadicalA('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)

N9 = RadicalB('N9', 1, TrpH_HFITs[8]*1e6)
H23 = RadicalB('H23', 1/2, TrpH_HFITs[22]*1e6)

RadicalA.add_all_to_simulation()
RadicalB.add_all_to_simulation()


# Here we convert from the single-radical eigenbasis used in the classes {RadicalA, RadicalB} to construct
# the relevant matrices in the total basis including both radicals (though if we only include nuclei in one radical,
# this essentially just simplifies to addiadding ang a direct product of 𝟙2 to account for the second (bare) electron !!!

# For a nucleus in Radical B: I_vec = 𝟙A ⊗ i_vec
def I_vec_total_basis_B(B_nucleus):
    return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalA.I_radical_dimension(), format='coo'), B_nucleus.I_vec_single_radical_basis()[x]) for x in [0,1,2]])

# For a nucleus in Radical A: I_vec = i_vec ⊗ 𝟙B
def I_vec_total_basis_A(A_nucleus):
    return np.array([scipy.sparse.kron(A_nucleus.I_vec_single_radical_basis()[x], scipy.sparse.identity(RadicalB.I_radical_dimension(), format='coo')) for x in [0,1,2]])


# For electron B: S_vec = 𝟙A ⊗ s_vec
def S_vec_total_basis_B(RadicalB):
    return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalA.I_radical_dimension(), format='coo'), RadicalB.S_vec_single_radical_basis()[x]) for x in [0, 1, 2]])

# For electron A: S_vec = s_vec ⊗ 𝟙B
def S_vec_total_basis_A(RadicalA):
    return np.array([scipy.sparse.kron(RadicalA.S_vec_single_radical_basis()[x], scipy.sparse.identity(RadicalB.I_radical_dimension(), format='coo')) for x in [0,1,2]])

# We calculate H_zee as -γBAS_vec where here S_vec = S_vec_A + S_vec_B

def H_zee_total_basis(field_strength, theta, phi):
    S_vec = S_vec_total_basis_A(RadicalA) + S_vec_total_basis_B(RadicalB)
    r_vec = np.array([np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi), np.cos(theta)])

    return (-gyromag / (2 * np.pi)) * field_strength * np.dot(r_vec, S_vec)

# Each term in the H_hfi Hamiltonian is of the form S_vec_A/b · Ai · I_vec_i

def H_hfi_total_basis():
    H = 0
    for nucleus in RadicalA.nuclei_included_in_simulation_A:
        H += sum(nucleus.hyperfine_interaction_tensor[i,j]*S_vec_total_basis_A(RadicalA)[i]@I_vec_total_basis_A(nucleus)[j] for i in range(3) for j in range(3))

    for nucleus in RadicalB.nuclei_included_in_simulation_B:
        H += sum(nucleus.hyperfine_interaction_tensor[i,j]*S_vec_total_basis_B(RadicalB)[i]@I_vec_total_basis_B(nucleus)[j] for i in range(3) for j in range(3))

    return H

# Can then calculate the full hamiltonian in the total basis

def Sparse_Hamiltonian_total_basis(field_strength, theta, phi, dipolar = False):
    H_tot = H_zee_total_basis(field_strength,theta,phi) + H_hfi_total_basis()
    if dipolar:
        H_tot += 0

    return H_tot

# Vmax then calculated as ever...

def Vmax(field_strength, theta, phi, display = False, display_eigenvalues = False, dipolar = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian_total_basis(field_strength, theta, phi, dipolar)

    if display:
        print(f'Sparse Hamiltonian created in {time.perf_counter() - startTime}s')
        print(f'Radical A: {RadicalA.nuclei_included_in_simulation_A}\n',f'Radical B: {RadicalB.nuclei_included_in_simulation_B}')

    valmax = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6    # Converting Vmax from Hz to MHz

    if display_eigenvalues:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}') # Showing the eigenvalues in rad s^-1

    if display:
        print(f'Vmax with {len(RadicalA.nuclei_included_in_simulation_A)+len(RadicalB.nuclei_included_in_simulation_B)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {time.perf_counter()-startTime}')
    return Vmax

def Densify(Sparse_Matrix):
    return Sparse_Matrix.todense()

eigenvalues, eigenvectors = np.linalg.eig(Densify(Sparse_Hamiltonian_total_basis(0.05, 1.0555, 4.1675)))
eigenvalues = np.real(np.sort(eigenvalues)[::-1])

print(eigenvalues/1e6)

alice_eigenvalues = np.sort(np.load('alice/all_eigenvalues.npy'))[::-1]
print(alice_eigenvalues[0]/ (2*np.pi*1e6))