from qiskit import *
from qiskit.providers.aer import *

from qiskit.extensions.simulator import Snapshot
from qiskit.extensions.simulator.snapshot import snapshot

from qiskit.quantum_info.random import random_unitary
from qiskit.extensions import *
from copy import deepcopy
import math
import numpy as np
import time
import random

#     reg["qubits"] = [1,3,5,6,9] , reg["coefs"] = [1,1,2,1] => [0,1,1,2,1,1,1,1,1]
#     reg["qubits"] = [3] , reg["coefs"] = [] => [0]*9
def expand_coefs(reg, n):
    j = 0
    coef = 0
    coefs = [0] * (n-1)
    for i in range(n-1):
        if i in reg["qubits"]:
            if i >= reg["qubits"][-1]:
                coef = 0
            else:
                coef = reg["coefs"][j]
            j = j + 1
        coefs[i] = coef
    return np.array(coefs, dtype = int)
    
#  0,1,2,3,4,5,6,7,8,9,10,11,12
# [1,1,1,1,2,1,1,2,1,1,2,1] , (i,j) = (2,6)
def add_entanglement(reg, i, j):
    i = reg["qubits"].index(i)
    j = reg["qubits"].index(j)
    m = len(reg["qubits"])
    added = np.array([0] * (m-1) , dtype = int)
    added[i:j] = [1] * (j-i)

    coefs = reg["coefs"] + added
    
    # make sure coefs are legal
    func1 = lambda x: np.minimum(x+1,m-(x+1))
    max_schmidt_rank = func1(np.arange(m-1))
#     print(max_schmidt_rank)
#     print(coefs)
    coefs = np.clip(coefs, 0, max_schmidt_rank)        
    
    for k in range(m-2):
        if  coefs[k+1] > coefs[k] + 1 :
            coefs[k+1] = coefs[k] + 1
    for k in range(m-2,0,-1):     
        if  coefs[k-1] > coefs[k] + 1 :
            coefs[k-1] = coefs[k] + 1
    reg["coefs"] = coefs
    return reg 

def Connect_two_registers(reg1, reg2, n):
    reg = {}
    reg["qubits"] = reg1["qubits"] + reg2["qubits"]
    reg["qubits"].sort()
    coefs_1 = expand_coefs(reg1,n) # to length = n
    coefs_2 = expand_coefs(reg2,n) # to length = n
    coefs = coefs_1 + coefs_2 # check legality (?)
    reg["coefs"] = [coefs[i] for i in reg["qubits"][:-1]]
        
    return reg

def sort_gates_by_registers(mini_list, regs):
    # when we can choose, we want t
    gates_list = []
    for (i,j) in mini_list:
        reg_i_index = [regs.index(reg) for reg in regs if i in reg["qubits"]][0]
        reg_j_index = [regs.index(reg) for reg in regs if j in reg["qubits"]][0] 
        if(reg_i_index == reg_j_index):
            gates_list.insert(0, (i,j))
        else:
            gates_list.append((i,j))
            
    return gates_list

def reorder_qc(qc,qregisters_dict):
    new_qc = []
    two_qubits_gates = []
# remove unwanted gates
    for gate in qc.data:
        if(type(gate[0]) is not (qiskit.extensions.standard.barrier.Barrier) and len(gate[1]) > 1 ):
            two_qubits_gates.append(gate)
   
    while (two_qubits_gates):
        gates = deepcopy(two_qubits_gates)
        mini_list = []
        available = list(range(n))
        for gate in gates:
            if (len(available) == 0):
                break
            i = gate[1][0][1] + qregisters_dict[gate[1][0][0].name]
            j = gate[1][1][1] + qregisters_dict[gate[1][1][0].name]
            (i, j) = sorted((i, j))
            if (i in available and j in available):
                available.remove(i)
                available.remove(j)
                mini_list.append((i, j))
                two_qubits_gates.remove(gate)
            elif (i in available): # and j not...
                available.remove(i)
            elif (j in available): # and i not...
                available.remove(j)
        new_qc.append(mini_list)
    return new_qc

# MPS simulation slows when entanglement grows, but entanglement requires superposition.
# Assumes no general U3,U2 gates, only Hadamards can cause superposition. I'm sure there is 
# The function returns the estimated number of none zero inputs of the state (if represented with a state vector) 
def super_position_estimation(qc,n):
    num_of_H = 0
    for gate in qc.data:
        if type(gate[0]) is (qiskit.extensions.standard.h.HGate):
            num_of_H = num_of_H + 1
            if num_of_H == n:
                return 2**num_of_H
        elif type(gate[0]) is (qiskit.extensions.unitary.UnitaryGate):
            return None
           
    estimated_superposition = 2**num_of_H
    return estimated_superposition

# Input: Quantum circuit.
# Output: 1. Array of length n-1 with the estimated number of schmidt coefficients for every bipartition 
#            If d is the maximal schmidt coefficient, running time is O(d^3)
#         2. Estimated_superposition (none zero inputs of the statenone zero inputs of the state)
# Example, for n = 6 qubits: The array can be [1,2,2,1,1]. There are 5 numbers, each one for a possible bipartiton of the qubits in 1-D. The maximal numbers can be [2,4,8,4,2].
# The superposition number can a number between 1 and 64(=2^n)
def mps_running_time_estimator(qc):
    qregisters_dict = {}
    offset = 0
    for qreg in qc.qregs:
        qregisters_dict[qreg.name] = offset
        offset = offset + qreg.size
    n = offset
### estimate superposition: 
    estimated_superposition = super_position_estimation(qc,n)
### estimate entanglement(schmidt coefficients):
    regs = [{"qubits": [i], "coefs": np.array([], dtype = int)} for i in range(n)]
    last_gates = [[] for i in range(n)]
    new_qc = reorder_qc(qc,qregisters_dict)

    for mini_list in new_qc:
        gates_list = sort_gates_by_registers(mini_list, regs)
        for (i, j) in gates_list:
            reg_i_index = [regs.index(reg) for reg in regs if i in reg["qubits"]][0]
            reg_j_index = [regs.index(reg) for reg in regs if j in reg["qubits"]][0]
            
            if(reg_i_index == reg_j_index): #same register, so their last_gates list isn't empty
                if(last_gates[i][-1] == j and last_gates[j][-1] == i):
                    continue
                reg = regs[reg_i_index]
                del regs[reg_i_index]
                same_register = True
                single_qubit_reg = False
            else:
                same_register = False
                single_qubit_reg =  (len(regs[reg_i_index]["qubits"]) == 1) or len(regs[reg_j_index]["qubits"]) == 1
                reg = Connect_two_registers(regs[reg_i_index], regs[reg_j_index], n)
                # order is important here, because we address registers by index, we muse delete the bigger index first  
                (reg_i_index,reg_j_index) = (min(reg_i_index,reg_j_index), max(reg_i_index,reg_j_index))
                del regs[reg_j_index] , regs[reg_i_index]
            

            expanding_right = (len(last_gates[i]) > 1 and 
                               last_gates[i][-2:] == sorted(last_gates[i][-2:]) and
                               last_gates[i][-2] > i and
                               last_gates[i][-1] < j and
                               len(last_gates[j]) < 2)
             
            expanding_left = (len(last_gates[j]) > 1 and 
                              last_gates[j][-2:] == sorted(last_gates[j][-2:], reverse = True) and
                              last_gates[j][-2] < j and 
                              last_gates[j][-1] > i and
                              len(last_gates[i]) < 2
                             )

            # Some heuristics of to treat a gate, depending on the current mini-registers of qubits and the previous gates 
            if (same_register):
                reg = add_entanglement(reg, i,j)
            elif (expanding_right):
                reg = add_entanglement(reg, last_gates[i][-1], j)
                reg = add_entanglement(reg, last_gates[i][-2],last_gates[i][-1])
            elif (expanding_left):
                reg = add_entanglement(reg, i, last_gates[j][-1])
                reg = add_entanglement(reg, last_gates[j][-1],last_gates[j][-2])
            elif ((len(set(last_gates[i]))  < 2 or len(set(last_gates[j])) < 2)) :
                reg = add_entanglement(reg, i,j)
            else:
                reg = add_entanglement(reg, i,j)
                reg = add_entanglement(reg, i,j)
                
            regs.append(reg)
            last_gates[i].append(j)
            last_gates[j].append(i)

    temp_reg = regs[0]
    for i in range(1,len(regs)):
        temp_reg = Connect_two_registers(temp_reg, regs[i], n)  
#     print(temp_reg["coefs"])
    
    estimated_schmidt_rank = [ 2**x for x in temp_reg["coefs"]]
 
    return estimated_schmidt_rank, estimated_superposition

#Running example:
n = 6
q = QuantumRegister(n)
c = ClassicalRegister(n)
qc = QuantumCircuit(q, c, name="circuit")
qubits_couples = [[1, 3],
                  [0, 4],
                  [1, 3],
                  [0, 1],
                  [0, 4]]
for couple in qubits_couples:
    U = random_unitary(4)   
    qc.unitary(random_unitary(4), couple)

(estimated_schmidt_rank, _) = mps_running_time_estimator(qc)
print("estimated_schmidt_rank = " + str(estimated_schmidt_rank))

import sys
sys.path.insert(0, "/gpfs/haifa/projects/q/qq/team/yotamvak/qiskit-terra/")
sys.path.insert(0, "/gpfs/haifa/projects/q/qq/team/yotamvak/qiskit-aqua/")

from tqdm import tqdm_notebook as tqdm
from qiskit import Aer
from qiskit.providers.aer import aerbackend
from qiskit.providers.aer.utils import qobj_utils
from qiskit.qobj import (QasmQobj, QobjExperimentHeader,
                         QasmQobjInstruction,
                         QasmQobjExperimentConfig,
                         QasmQobjExperiment,
                         QasmQobjConfig)
#from qiskit.qobj.models.qasm import QasmQobjInstruction, QasmQobjExperiment
from qiskit.qobj.models.base import QobjHeader
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import copy
from copy import deepcopy
from pprint import pprint

from qiskit.assembler import assemble_circuits, run_config
from qiskit.assembler.run_config import RunConfig

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np
from tqdm import tqdm_notebook as tqdm
from qiskit.chemistry import QiskitChemistry
from qiskit import *
from qiskit.providers.aer import * 
from qiskit.aqua import QuantumInstance
backend = Aer.get_backend("qasm_simulator")
tensor_qi = QuantumInstance(backend, backend_options = {"method": "tensor_network"})
statevector_qi = QuantumInstance(backend, backend_options = {"method": "statevector"})
from time import time
from qiskit.aqua.components.variational_forms import RYRZ
import matplotlib.pyplot as plt

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
# driver = PySCFDriver(atom='H .0 .0 .0; N .0 .0 1; H 0 0 2; H 0 1 1; H 0 -1 1; H 0 -1 2',
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5',
# driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 1;',
                    unit=UnitsType.ANGSTROM,
                    basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubitOp = ferOp.mapping(map_type)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles)
num_qubits = qubitOp.num_qubits
print(num_qubits)


ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
operators = ferOp.mapping(map_type)
operators = operators.two_qubit_reduced_operator(num_particles)
num_qubits = qubitOp.num_qubits

from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
init_state = HartreeFock(num_qubits, num_spin_orbitals, num_particles)

from qiskit.aqua.components.optimizers import L_BFGS_B, NELDER_MEAD,COBYLA
optimizer = NELDER_MEAD(100, 100)

from qiskit.aqua.components.variational_forms import RYRZ
index = 0
class _RYRZ(RYRZ):
    def construct_circuit(*args, **kwargs):
        global index
        print(index, end="\t")
        index += 1
        return RYRZ.construct_circuit(*args, **kwargs)
var_form = RYRZ(num_qubits, initial_state=init_state)

# instructions = []

# final_val = 0
# new_instr_1 = QasmQobjInstruction(name="snapshot",
#                             type= "expectation_value_matrix",
#                             label= "energy",
#                             qubits=range(num_qubits),
#                             params= [[[1,0], [[list(range(num_qubits)), qubitOp.matrix.toarray()]]]])

# instructions.append(new_instr_1)
    
bounds = var_form.parameter_bounds
low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
initial_point = np.random.uniform(low, high)

qc = var_form.construct_circuit(initial_point)



