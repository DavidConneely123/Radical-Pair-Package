import numpy as np
import pandas as pd

from RadicalPair import *


# A simple function to transform a dense matrix into a sparse one

def Densify(Sparse_Matrix):
    return Sparse_Matrix.todense()

def Transform_to_eigenbasis(Matrix_of_eigenvectors, Matrix_to_transform):
    A = Matrix_of_eigenvectors
    A_inv = Matrix_of_eigenvectors.conj().T    # NB; if matrix is Hermitian then its inverse is its conjugate transpose

    return A_inv @ Matrix_to_transform @ A

def Plot_Spectrum(eigenvalues1, eigenvalues2 = None):
    fig, ax = plt.subplots()
    if eigenvalues2 is None:
        x = np.linspace(1, 2, 100)
        y = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x, y, color='red')

    else:
        x1 = np.linspace(1, 2.5, 100)
        y1 = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x1, y1, color='red')

        x2 = np.linspace(2.5, 4, 100)
        y2 = np.linspace(eigenvalues2, eigenvalues2, 100)
        ax.plot(x2, y2, color='black')

def Histogram_Height(eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, Centre_Frequency, Frequency_Bin_Width = 0.5):
    if Centre_Frequency == 0.25:
        Histogram_Height = 0

    else:
        Histogram_Height = 0

        number_of_states = len(eigenvalues_ascending_order_MHz)

        for i in range(number_of_states):
            for j in range(number_of_states):
                    if Centre_Frequency - 0.5*Frequency_Bin_Width <= np.abs(eigenvalues_ascending_order_MHz[i] - eigenvalues_ascending_order_MHz[j]) <= Centre_Frequency + 0.5*Frequency_Bin_Width:
                        Histogram_Height += np.abs(Populations[i] - Populations[j])*np.square(np.abs(Transition_Hamiltonian[i, j]))   # [i, number_of_states-j-1]


    return Histogram_Height

def Action_Spectrum(eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, Maximum_Frequency, display = False):
    # We can use multiprocessing to calculate the different histogram heights all in parallel, giving large speed up to computation
    Vmax = max(eigenvalues_ascending_order_MHz) - min(eigenvalues_ascending_order_MHz)

    Histogram_Heights = []
    for v in np.linspace(0.25, Maximum_Frequency+0.25, 2*Maximum_Frequency+1):  #[0.25, 0.75, 1.25, ..., 100.25]
        Histogram_Heights.append(Histogram_Height(eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, v))


    if display:
        fig1,ax1 = plt.subplots()
        ax1.bar(np.linspace(0.25, Maximum_Frequency+0.25, 2*Maximum_Frequency+1), Histogram_Heights, width=0.5, align = 'center', color='green')
        ax1.plot(np.linspace( np.ceil(Vmax), np.ceil(Vmax)), np.linspace(0, 1.1*max(Histogram_Heights)), color ='red')
    return Histogram_Heights

def Number_of_Energy_Gaps(eigenvalues_ascending_order_MHz, Centre_Frequency):
        Number_of_Gaps = 0
        number_of_states = len(eigenvalues_ascending_order_MHz)

        for i in range(number_of_states):
            for j in range(number_of_states):
                if Centre_Frequency - 0.5 * 0.5 <= np.abs(eigenvalues_ascending_order_MHz[i] - eigenvalues_ascending_order_MHz[j]) <= Centre_Frequency + 0.5 * 0.5:
                    Number_of_Gaps += 1

        return Number_of_Gaps

def Graph_of_Energy_Gaps(eigenvalues_ascending_order_MHz, Maximum_Frequency):
    fig2, ax2 = plt.subplots()

    Bar_Heights = []
    for v in np.linspace(0.25, Maximum_Frequency + 0.25, 2 * Maximum_Frequency + 1):
        Bar_Heights.append(Number_of_Energy_Gaps(v))

    ax2.bar(np.linspace(0.25, Maximum_Frequency + 0.25, 2 * Maximum_Frequency + 1), Bar_Heights, width=0.5, align='center', color='green')



def Combine_subsystem_eigenvalues(eigenvalues_subsystem_a, eigenvalues_subsystem_b):
    sums_of_eigenvalues = []
    for eigenvalue2 in eigenvalues_subsystem_a:
        for eigenvalue1 in eigenvalues_subsystem_b:
            sums_of_eigenvalues.append(eigenvalue2 + eigenvalue1)

    return np.sort(sums_of_eigenvalues)

def Compare_Spectra_Exact_v_Subsystem():
    RadicalA.add_all_to_simulation()

    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, 0, 0))
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh() returns eigenvalues sorted in ASCENDING order, with the COLUMN eigenvectors[:,i] corresponding to eigenvalue w[i]
    eigenvectors = eigenvectors.T

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in RadicalA.all_A:
        if nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    Hamiltonian_subsystem_b = Densify(Sparse_Hamiltonian_total_basis(0.05, 0, 0))
    eigenvalues_b, eigenvectors_b = np.linalg.eigh(Hamiltonian_subsystem_b)
    eigenvectors_b = eigenvectors_b.T

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in RadicalA.all_A:
        if not nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    Hamiltonian_subsystem_a = Densify(Sparse_Hamiltonian_total_basis(0.0, 0, 0))
    eigenvalues_a, eigenvectors_a = np.linalg.eigh(Hamiltonian_subsystem_a)
    eigenvectors_a = eigenvectors_a.T

    eigenvalues_subsystem_method = Combine_subsystem_eigenvalues(eigenvalues_a, eigenvalues_b)

    Plot_Spectrum(eigenvalues, eigenvalues2=eigenvalues_subsystem_method)

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()


    return eigenvalues, eigenvalues_subsystem_method

def tensor_list(number_of_tensors, axiality, off_diagonals = False, not_random = False):

    list_of_tensors = []

    for _ in range(number_of_tensors):
        tensor = np.zeros((3,3))
        tensor[0,0] = random.uniform(-(1-axiality), (1-axiality))*10
        tensor[1,1] = random.uniform(-(1-axiality), (1-axiality))*10
        tensor[2,2] = random.uniform(-1,1)*10

        if off_diagonals:
            tensor[0,1] = random.uniform(-(1-axiality), (1-axiality))*0.2
            tensor[1,0] = random.uniform(-(1-axiality), (1-axiality))*0.2

            tensor[1, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 1] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

            tensor[0, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 0] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

        if not_random:
            tensor[0,0] = 1-axiality
            tensor[1,1] = 1-axiality
            tensor[2,2] = axiality


        tensor = np.multiply(tensor, 1e6)
        list_of_tensors.append(tensor)




    return list_of_tensors

def Histogram_Heights_Single_Field_Direction(theta,phi, display = False):
    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, phi))

    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh gives [w[i]] eigenvalues in ascending order and a matrix with eigenvectors as COLUMNS not ROWS !!!
    eigenvalues = eigenvalues / 1e6   # Converting Eigenvalues to MHz

    Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors
    Populations = np.diag(Singlet_Projection_Matrix)
    Transition_Hamiltonian = eigenvectors.conj().T @ H_perp_total_basis(1e-5, theta, phi) @ eigenvectors

    if display:
        Histogram_Heights_Single_Field_Direction = Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,100, display = True)

    else:
        Histogram_Heights_Single_Field_Direction = Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian, 100)

    return Histogram_Heights_Single_Field_Direction

def ASH_Averaged_Over_Field_Directions(Maximum_Frequency, Number_of_Field_Directions, Save_Histogram_Heights = False, File_Name = None):
    Cumulative_Histogram_Heights = np.zeros((201,))
    for [theta, phi] in [list_of_field_directions[i] for i in range(Number_of_Field_Directions)]:
        Cumulative_Histogram_Heights += Histogram_Heights_Single_Field_Direction(theta, phi)
        print(f'Completed: Theta = {theta} , Phi =  {phi}')

    fig1, ax1 = plt.subplots()
    ax1.bar(np.linspace(0.25, Maximum_Frequency + 0.25, 2 * Maximum_Frequency + 1), Cumulative_Histogram_Heights,width=0.5, align='center', color='green')

    if Save_Histogram_Heights:
        df = pd.DataFrame({"Histogram Heights": Cumulative_Histogram_Heights})
        df.to_csv(File_Name)

def Visualise_Matrix(Matrix):
    thing = np.array([Matrix[i, j] for i in range(int(np.sqrt(np.size(Matrix)))) for j in range(int(np.sqrt(np.size(Matrix))))])


    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    _x = np.arange(np.sqrt(np.size(Matrix)))
    _y = np.arange(np.sqrt(np.size(Matrix)))

    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = thing
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Matrix Elements')






