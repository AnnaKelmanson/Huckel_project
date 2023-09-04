##################
# Anna Borisova, Lorenzo Gramolini, Erasmus Mundus program in Theoretical Chemistry and Computational modeling, Projet informatique cours
#################
import math
import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from pathlib import Path

#constants
CC = 2.46 #instead of using angle I took length of the triangle side in front of the angle 120 deg
rLayers = 1.42  #distance beteween carbons
z = 0.0 #Z set 0
#functions

def layer(rings,X1,Y1): #just setting dots on a line with given interval
    coordinates1 = []
    for x in range(rings+1):
        coordinates1.append("C" + "    " + '%.9f' %X1 + "    " + '%.9f' %Y1 + "    " + '%.9f' %z)
        X1 += CC
    return(coordinates1)

def graphene(rings,layers): 
    output = []
    X1 = 0
    X2 = X1 - CC*0.5
    Y1 = 0
    Y2 = Y1 - math.sqrt(rLayers**2-(CC*0.5)**2)
    for i in range(layers):
        output.append(layer(rings,X1,Y1))
        output.append(layer(rings,X2,Y2))
        output.append(layer(rings,X2,Y2-rLayers))
        output.append(layer(rings,X1,Y2-rLayers-math.sqrt(rLayers**2-(CC*0.5)**2)))
        Y1 -= math.sqrt(rLayers**2-(CC*0.5)**2)*2 + 2*rLayers
        Y2 -= math.sqrt(rLayers**2-(CC*0.5)**2)*2 + 2*rLayers
    output = output[:layers*2+2] # crutches
    if num_of_y_rings % 2 == 1:
        del output[0][-1]
        del output[-1][-1]
    elif num_of_y_rings % 2 == 0:
        del output[0][-1]
        del output[-1][0]
    return((list(itertools.chain.from_iterable(output))))

# user interactive
num_of_x_rings = int(input("Number of rings in X-axis = "))
num_of_y_rings = int(input("Number of rings in Y-axis = "))
output = graphene(num_of_x_rings,num_of_y_rings)
# writing xyz file
natom = len(output)

with open(f"graphene-C{natom}.xyz", "w") as outFile:
    outFile.write(f"{natom}\n")
    outFile.write(f"Graphene C{natom} in xyz format\n")
    outFile.write("\n".join(output))


def read_xyz_file(natom):
    with open(f"graphene-C{natom}.xyz", "r") as xyz_file:
        lines = xyz_file.readlines()
    
    xyz = [line.split() for line in lines if len(line.split()) == 4]
    xyz2 = copy.deepcopy(xyz)
    for i in range(len(xyz2)):
        xyz2[i].insert(0, i + 1)
    
    return xyz2

def calculate_distances(xyz2):
    distances = []
    for i in range(len(xyz2)):
        for j in range(i, len(xyz2)):
            distance = math.sqrt(
                (float(xyz2[i][2]) - float(xyz2[j][2])) ** 2 +
                (float(xyz2[i][3]) - float(xyz2[j][3])) ** 2 +
                (float(xyz2[i][4]) - float(xyz2[j][4])) ** 2
            )
            if distance != 0:
                element = [xyz2[i][1], xyz2[j][1], xyz2[i][0], xyz2[j][0], distance]
                distances.append(element)
    return distances


def find_couples(distances):
    couples = []
    for i in range(len(distances)):
        if round(distances[i][4], 3) == round(1.42, 3):
            couples.append(distances[i])
    return couples


def build_h_matrix(xyz2, couples, alfa, beta):
    H = []
    for i in range(len(xyz2)):
        row = []
        H.append(row)

    for i in range(len(xyz2)):
        for j in range(len(xyz2)):
            H[i].append(0)

    for i in range(len(xyz2)):
        H[i][i] = float(alfa)

    for i in range(len(couples)):
        H[couples[i][2] - 1][couples[i][3] - 1] = float(beta)

    for i in range(len(xyz2)):
        for j in range(len(xyz2)):
            if i != j:
                if H[i][j] != 0:
                    H[j][i] = H[i][j]

    return H


def calculate_eigenvalues_and_eigenvectors(H):
    Hm = np.array(H)
    [E, V1] = np.linalg.eig(Hm)
    idx = E.argsort()[::1]
    E = E[idx]
    V1 = V1[:, idx]
    V = np.transpose(V1)
    V = V.tolist()
    return E, V


def write_results_to_file(natom, xyz2, E, V):
    filename = f"graphene-C{natom}.txt"
    filename2=f"graphene-C{natom}.xyz"
    with open(filename, "w") as out:
        for i, energy in enumerate(E):
            orbital_info = [[0, 0, 0] for _ in range(len(V))]

            for j in range(len(orbital_info)):
                orbital_info[j][2] = "pz"
                orbital_info[j][1] = V[i][j]
                orbital_info[j][0] = xyz2[j][0]

            orbital_values = [abs(item[1]) for item in orbital_info]
            orbital_values.sort()
            normalized_info = copy.deepcopy(orbital_info)
            norm = 1 / orbital_values[-1]

            for item in normalized_info:
                item[1] *= norm

            out.write("Energy={:.2f}\n".format(energy))
            out.write("|psi>=+ ")
            for item in orbital_info:
                out.write("({:.2f})|{}>+ ".format(item[1], item[0]))
            out.write("\n\n\nPLEASE COPY AND PASTE THE FOLLOWING LINES ON THE JMOL CONSOLE TO VISUALIZE THE MOLECULAR ORBITAL\n")
            out.write("load {}/{}\n".format(Path.cwd(), filename2))
            out.write("select all; wireframe on; spacefill off\n")
            for item, normalized_item in zip(orbital_info, normalized_info):
                if item[1] > 0:
                    out.write('select(atomno={});lcaoCartoon delete color {} {} scale {} create "{}"\n'.format(
                        item[0], "red", "blue", abs(round(normalized_item[1], 3)), item[2]))
                elif item[1] < 0:
                    out.write('select(atomno={});lcaoCartoon delete color {} {} scale {} create "{}"\n'.format(
                        item[0], "blue", "red", abs(round(normalized_item[1], 3)), item[2]))

            out.write("\n\n\n\n")


def plot_energy_diagram(E):
    index = np.arange(1, len(E) + 1)
    x = np.array(index)
    y = np.array(E)

    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(E[0] - 0.5, E[-1] + 0.5, 0.5))
    ax.set_ylim(E[0] - 0.5, E[-1] + 0.5)
    ax.set_xlim(0, len(index) + 1)
    ax.set_xlabel('# orbital')
    ax.set_ylabel('Energy')

    plt.title("graphene-C" + str(natom) + "")
    for i in range(len(x)):
        plt.scatter(x[i], y[i], label=round(E[i], 3), marker='*', s=30)
    plt.show()


if __name__ == '__main__':
    xyz2 = read_xyz_file(natom)
    distances = calculate_distances(xyz2)
    couples = find_couples(distances)
    H = build_h_matrix(xyz2, couples, alfa=0, beta=-1)
    E, V = calculate_eigenvalues_and_eigenvectors(H)
    write_results_to_file(natom, xyz2, E, V)
    plot_energy_diagram(E)

