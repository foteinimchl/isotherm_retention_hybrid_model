# %%
import numpy as np
import pandas as pd
from torch import nn
import scipy.integrate as scp
import matplotlib.pyplot as plt
from keras.models import load_model

# ANN Model loading
import torch
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

import pickle # Quick saving format

def save_pkl(data, save_name):
    """
    Saves .pkl file from of data in folder: tmp/
    """
    with open(save_name, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)
    print(f'File saved at: {save_name}')
    return None

def load_pkl(file_name):
    """
    Loads .pkl file from path: file_name
    """
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    print(f'Data loaded: {file_name}')
    return data

def ReLUNet(n_input, n_layers, n_hidden, n_output):
    """
    Initiate a ReLU feed forward neural network based on the hyperparams:
    n_input, n_layers, n_hidden, n_output
    """
    layers = []

    # Append input layer
    layers.append(nn.Linear(n_input, n_hidden))
    layers.append(nn.ReLU())

    # Append hidden layers
    for i in range(n_layers):
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.ReLU())

    # Append output layers
    layers.append(nn.Linear(n_hidden, n_output))
    return nn.Sequential(*layers)

class ReLUNet_inference():
    """
    Class for inference of ANNs. The minmax normalization is loaded into the class.
    """
    def __init__(self, m, normP):
        """
        Initialize class for ANN inference
        """
        self.m = m.cpu()
        self.normP = normP
    def __call__(self, x):
        """
        Inference function, converts x into normalized value and then a tensor.
        Then converts output to real value and numpy array.
        """
        m = self.m
        normP = self.normP
        x = self.mm_norm(x, normP[0])
        
        x = torch.tensor(np.array(x, dtype = 'float64')).cpu()
        x = m(x).cpu().detach().numpy()
        x = self.mm_rev(x, normP[1])
        return x
    
    def mm_norm(self, arr, normP):
        """
        Min max normalisation
        """
        arrmax = normP[0] 
        arrmin = normP[1]
        return (arr - arrmin)/(arrmax - arrmin)

    def mm_rev(self, norm, normP):
        """
        Reverse min max normalisation
        """
        arrmax = normP[0] 
        arrmin = normP[1]
        return norm*(arrmax - arrmin) + arrmin

def load_model(fn):
    """
    Load ANN model for inference
    """

    P = load_pkl(fn)
    SD = P['state_dict']
    structure = P['structure']
    normP = P['normP']

    m = ReLUNet(*structure)
    m.load_state_dict(SD)
    m = ReLUNet_inference(m, normP)
    return m

def mm_norm(arr, normP):
    """
    Min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return (arr - arrmin)/(arrmax - arrmin)

def mm_rev(norm, normP):
    """
    Reverse min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return norm*(arrmax - arrmin) + arrmin

# %%
class MCSGP():
    def __init__(self, params):
        """
        MCSGP column simulation model
        """
        self.params = params
        return None

    def model(self, t, x, params, concentrations_only = False):
        
        x = x.copy()
        # All variables in state are non negative
        x[x <= 0] = 0 # clip all negative values to 0

        # ----- Unpack parameters ----- #
        dz = params['dz']

        # Process Params

        CFEED1 = params['CFEED1']
        CFEED2 = params['CFEED2']
        CFEED3 = params['CFEED3']
        CFEED4 = params['CFEED4']
        Q = params['Q']
        Q_I1 = params['Q_I1']
        Q_I2 = params['Q_I2']
        Q_B = params['Q_B']
        Cmod1 = params['Cmod1']
        Cmod2 = params['Cmod2']

        HMOD = params['HMOD']
        A1_1 = params['A1_1']
        A1_2 = params['A1_2']
        A1_3 = params['A1_3']
        A1_4 = params['A1_4']
        A2_1 = params['A2_1']
        A2_2 = params['A2_2']
        A2_3 = params['A2_3']
        A2_4 = params['A2_4']
        A3_1 = params['A3_1']
        A3_2 = params['A3_2']
        A3_3 = params['A3_3']
        A3_4 = params['A3_4']
        A4_1 = params['A4_1']
        A4_2 = params['A4_2']
        A4_3 = params['A4_3']
        A4_4 = params['A4_4']
        A5_1 = params['A5_1']
        A5_2 = params['A5_2']
        A5_3 = params['A5_3']
        A5_4 = params['A5_4']
        A6_1 = params['A6_1']
        A6_2 = params['A6_2']
        A6_3 = params['A6_3']
        A6_4 = params['A6_4']
        A7_1 = params['A7_1']
        A7_2 = params['A7_2']
        A7_3 = params['A7_3']
        A7_4 = params['A7_4']
        A8_1 = params['A8_1']
        A8_2 = params['A8_2']
        A8_3 = params['A8_3']
        A8_4 = params['A8_4']

        z_size = int(np.round(1/dz)) + 1
        z = np.linspace(0, 1, z_size)
        z = np.round(z*100)/100
        
        # Unpack x (state)
        if len(x.shape) == 1:   # If state is one dimensional 
            # (during normal simulation)
            C1 = x[        :  z_size]
            C2 = x[  z_size:2*z_size]
            C3 = x[2*z_size:3*z_size]
            C4 = x[3*z_size:4*z_size]
            Int_C1 = x[4*z_size:5*z_size]
            Int_C2 = x[5*z_size:6*z_size]
            Int_C3 = x[6*z_size:7*z_size]
            Int_C4 = x[7*z_size:8*z_size]
        else:
            print('Warning, dimension of state (x) is not 1D')

        # ----------------------------- Start of Equations ----------------------------- #
        # ----- Algebraic ----- # (Vectorized calculation)

        zero_eps = 1e-10 # Prevent division by zero
        
        H0_1 = A1_1*(C1**A2_1)  
        H0_2 = A1_2*(C1**A2_2)
        H0_3 = A1_3*(C1**A2_3)
        H0_4 = A1_4*(C1**A2_4)

        # H2 = np.zeros(C.shape)
        H2_1 = A3_1*(H0_1**A4_1)
        H2_2 = A3_2*(H0_2**A4_2)
        H2_3 = A3_3*(H0_3**A4_3)
        H2_4 = A3_4*(H0_4**A4_4)

        # H1 = np.zeros(C.shape)
        H1_1 = H0_1 - H2_1
        H1_2 = H0_2 - H2_2
        H1_3 = H0_3 - H2_3
        H1_4 = H0_4 - H2_4

        # Saturation capacity of site 1
        SC1_1 = A5_1*(H1_1**A6_1)     
        SC1_2 = A5_2*(H1_2**A6_2)  
        SC1_3 = A5_3*(H1_3**A6_3)  
        SC1_4 = A5_4*(H1_4**A6_4)  

        # Saturation capacity of site 2
        SC2_1 = A7_1*(H2_1**A8_1) 
        SC2_2 = A7_2*(H2_2**A8_2)  
        SC2_3 = A7_3*(H2_3**A8_3)  
        SC2_4 = A7_4*(H2_4**A8_4)  

        # # Solid Phase Equilibrium Concentration
        SPCE1 = HMOD*C1
        SPCE2 = C2*(H1_2/(1 + (C2*H1_2/SC1_2+C3*H1_3/SC1_3+C4*H1_4/SC1_4)) + H2_2/(1 + (C2*H2_2/SC2_2+C3*H2_3/SC2_3+C4*H2_4/SC2_4)))
        SPCE3 = C3*(H1_3/(1 + (C2*H1_2/SC1_2+C3*H1_3/SC1_3+C4*H1_4/SC1_4)) + H2_3/(1 + (C3*H2_2/SC2_2+C3*H2_3/SC2_3+C4*H2_4/SC2_4)))
        SPCE4 = C4*(H1_4/(1 + (C2*H1_2/SC1_2+C3*H1_3/SC1_3+C4*H1_4/SC1_4)) + H2_4/(1 + (C4*H2_2/SC2_2+C3*H2_3/SC2_3+C4*H2_4/SC2_4)))

        # Unpack x (state)
        if len(x.shape) == 1:   # If state is one dimensional 

            # 1 ANN for all components 
            x_Cout = [[CFEED1, CFEED2, CFEED3, CFEED4, Cmod1, Cmod2, Q_I1, Q_I2, Q_B, 
                    C1[0], C2[0], C3[0], C4[0], SPCE1[-1], SPCE2[-1], SPCE3[-1], SPCE4[-1]]]
            Cout_res = Cout_ANN(x_Cout)
            Cout_res = Cout_res[0]
            Cout_res[Cout_res<0] = 0
            C1[-1] = Cout_res[0]
            C2[-1] = Cout_res[1]
            C3[-1] = Cout_res[2]
            C4[-1] = Cout_res[3]
            Int_C1[-1] += C1[-1]
            Int_C2[-1] += C2[-1]
            Int_C3[-1] += C3[-1]
            Int_C4[-1] += C4[-1]
          
        else:
            print('Warning, dimension of state (x) is not 1D')
        
        # Calculation of mass for each component
        m_1 = C1*Q
        m_2 = C2*Q
        m_3 = C3*Q
        m_4 = C4*Q
        Pur = C3/(C2+C3+C4+zero_eps)
            

         # ------------------------------ End of Equations ------------------------------ #


        # --------------- Pack up outputs --------------- #

        if concentrations_only:
            return np.concatenate([C1, C2, C3, C4])
        
        # If not grads only, back calculation with state trajectory (X) and output all
        else:

            # Create dataframe based on the time
            results = pd.DataFrame([t], index = ['Time']).T

            # Parameters
            for k in list(params.keys()):
                if k in ['t0', 'tF']:
                    pass
                else:
                    results[k] = params[k]
            
            # Purity Outputs    
            Purity_labels = []
            for vn in ['Pur']:
                Purity_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            Purity_df = pd.DataFrame(Pur, index = Purity_labels).T
            results = pd.concat([results, Purity_df], axis = 1)
            
            # MASS Outputs
            MASS = np.concatenate([m_1, m_2, m_3, m_4])
            MASS_labels = []
            for vn in ['m_1', 'm_2', 'm_3', 'm_4']:
                MASS_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            MASS_df = pd.DataFrame(MASS, index = MASS_labels).T
            results = pd.concat([results, MASS_df], axis = 1)
            
            # Henry Constants Outputs        
            HC = np.concatenate([H0_1, H0_2, H0_3, H0_4, 
                                 H2_1, H2_2, H2_3, H2_4, 
                                 H1_1, H1_2, H1_3, H1_4])        
            
            HC_labels = []
            for vn in ['H0_1', 'H0_2', 'H0_3', 'H0_4', 
                       'H2_1', 'H2_2', 'H2_3', 'H2_4', 
                       'H1_1', 'H1_2', 'H1_3', 'H1_4']:
                HC_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            HC_df = pd.DataFrame(HC, index = HC_labels).T
            results = pd.concat([results, HC_df], axis = 1)

            # Saturation Capacity Outputs
            SC = np.concatenate([SC1_1, SC1_2, SC1_3, SC1_4, 
                                 SC2_1, SC2_2, SC2_3, SC2_4])
            SC_labels = []
            for vn in ['SC1_1', 'SC1_2', 'SC1_3', 'SC1_4', 
                       'SC2_1', 'SC2_2', 'SC2_3', 'SC2_4']:
                SC_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            SC_df = pd.DataFrame(SC, index = SC_labels).T
            results = pd.concat([results, SC_df], axis = 1)

            # Solid Phase Equilibrium Concentrations Outputs
            SPE = np.concatenate([SPCE1, SPCE2, SPCE3, SPCE4])
            SPE_labels = []
            for vn in ['SPCE1', 'SPCE2', 'SPCE3', 'SPCE4']:
                SPE_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            SPE_df = pd.DataFrame(SPE, index = SPE_labels).T
            results = pd.concat([results, SPE_df], axis = 1)

            # Liquid Phase Concentrations Outputs
            LPC_Int = np.concatenate([C1, C2, C3, C4, 
                                      Int_C1, Int_C2, Int_C3, Int_C4])
            LPC_Int_labels = []
            for vn in ['C1', 'C2', 'C3', 'C4', 
                       'Int_C1', 'Int_C2', 'Int_C3', 'Int_C4']:
                LPC_Int_labels += [f'{vn}' + f'_{i}' for i in range(z_size)]
            LPC_Int_df = pd.DataFrame(LPC_Int, index = LPC_Int_labels).T
            results = pd.concat([results, LPC_Int_df], axis = 1)

            return results, LPC_Int
            
    def simulation(self, x0, params = 'def'):
        """
        Simulation of the model using scipy.integrate.solve_ivp 
        with the self.ode_method method (def: BDF)
        """
        m = self.model
        # Update parameters if new params is in the argument
        if params != 'def':
            self.params.update(params)

        # Unpack model params
        params = self.params.copy()
        t0 = params['t0']
        tF = params['tF']
        dt = params['dt']
        dz = params['dz']
        C_PUMP_grad = params['C_PUMP_grad']
        
        # Create time and z space mesh
        steps = int(np.round((tF - t0)/dt))
        t = np.linspace(t0, tF, steps + 1)
        z_size = int(np.round(1/dz)) + 1
        z = np.linspace(0, 1, z_size)
        z = np.round(z*100)/100

        # Prepare recording arrays
        X = np.zeros((x0.size, t.size))
        X[:, 0] = x0
        results_record = []

        # Loop over the time trajectory t
        # updating each x0[0] with C_PUMP_grad and the rest of x0 with the output
        for i in range(len(t) - 1):
            # Update time horizon of simulation
            t0 = t[i]
            tF = t[i + 1]
            
            params.update({'t0': t0, 'tF': tF})
            # Simulate the model
            result_list, output = m(t0, x0, params)

            # All variables in state are non negative
            output_y = output.copy()
            output_y[output_y <= 0] = 0 # clip all negative values to 0
            # Update initial state x0 (current state)
            x0 = output_y
            x0[0] += C_PUMP_grad*dt
            X[:, i + 1] = x0
            results_record.append(result_list)
        
        results_record.append(result_list)

        results = pd.concat(results_record)

        # ------ Pack up outputs ------ #
        outputs = {'r': results, 'last_state': X[:, -1], 'last_t': t[-1], 'o': output}
        return outputs
    
    def simulation_interconnected(self, x0, t, C_t1, C_t2, C_t3, C_t4, params = 'def'):
        """
        Special function to simulate interconnected columns
        x0 is initial state, t is the time trajectory, c_t is the inlet 
        cocentration trajectory
        """
        m = self.model
        # Update parameters
        if params != 'def':
            self.params.update(params)

        # Unpack model params
        params = self.params.copy()
        dz = params['dz']
        z_size = int(np.round(1/dz)) + 1
        z = np.linspace(0, 1, z_size)
        z = np.round(z*100)/100

        # Initial state based on concetration trajectory
        x0[0] = C_t1[0]
        x0[z_size] = C_t2[0]
        x0[2*z_size] = C_t3[0]
        x0[3*z_size] = C_t4[0]

        # Prepare recording arrays
        X = np.zeros((x0.size, t.size))
        X[:, 0] = x0
        results_record = []
        
        # Loop over the time trajectory t
        # updating each x0[0] with c_t and the rest of x0 with the output
        for i in range(len(t) - 1):
            # Update time horizon of simulation
            t0 = t[i]
            tF = t[i + 1]
            # Update the params dictionary
            params.update({'t0': t0, 'tF': tF})
            # Simulate the model
            result_list, output = m(t0, x0, params)

            # All variables in state are non negative
            output_y = output.copy()
            output_y[output_y <= 0] = 0 # clip all negative values to 0
            # Update initial state x0 (current state)
            x0 = output_y
            x0[0] = C_t1[i + 1]
            x0[z_size] = C_t2[i + 1]
            x0[2*z_size] = C_t3[i + 1]
            x0[3*z_size] = C_t4[i + 1]
            X[:, i + 1] = x0
            results_record.append(result_list)
        
        results_record.append(result_list)

        results = pd.concat(results_record)

        # ------ Pack up outputs ------ #
        outputs = {'r': results, 'last_state': X[:, -1], 'last_t': t[-1], 'o': output}
        return outputs
    
    def one_cycle(self, last_t, last_state = None, params = 'def'):
        """
        Simulation of one cycle of the process
        last_state is a list of arrays containing x0 of each columns [x0_1, x0_2]
        If last_state is None, then both columns start empty
        """
        # ----- MCSGP Process Schedule ----- #
        # Twin column chromatography

        # Update parameters
        if params != 'def':
            self.params.update(params)
        # Unpack params
        params = self.params.copy()
        
        # ----- Unpack parameters ----- #
        dz = params['dz']
        z_size = int(np.round(1/dz)) + 1

        # Process Schedule Params
        CFEED1 = params['CFEED1']
        CFEED2 = params['CFEED2']
        CFEED3 = params['CFEED3']
        CFEED4 = params['CFEED4']
        Q = params['Q']
        Q_I1 = params['Q_I1']
        Q_B = params['Q_B']
        Q_I2 = params['Q_I2']
        Q_col_2 = params['Q_col_2']
        Cmod1 = params['Cmod1']
        Cmod2 = params['Cmod2']
        C_PUMP_grad_I1 = params['C_PUMP_grad_I1']
        C_PUMP_grad_B1 = params['C_PUMP_grad_B1']
        C_PUMP_grad_I2 = params['C_PUMP_grad_I2']
        C_PUMP_grad_B2 = params['C_PUMP_grad_B2']
        C_PUMP_grad_B2_col_2 = params['C_PUMP_grad_B2_col_2']
        T_I1 = params['T_I1']
        T_B1 = params['T_B1']
        T_I2 = params['T_I2']
        T_B2 = params['T_B2']

        # Initial state of columns - Specify values of equilibrated columns
        if last_state == None:

            x0_1 = np.zeros(8*z_size) # Values for inlet and outlet modifier concentration of column 1
            x0_1[0] = val
            x0_1[1] = val
            x0_2 = np.zeros(8*z_size) # Values for inlet and outlet modifier concentration of column 2
            x0_2[0] = val
            x0_2[1] = val

        else:
            x0_1 = last_state[0]
            x0_2 = last_state[1]

        # Prepare recording list
        o1 = [] # Record outputs of column 1
        o2 = [] # Record outputs of column 2

        
        # ---------------------------------- Step I1S1 ---------------------------------- #
        # Interconnected  (right column is column 1)
        # Simulate Column 1
        Qin_1 = Q_I1
        Cin1_1 = Cmod1
        Cin1_2 = 0
        Cin1_3 = 0
        Cin1_4 = 0
        Int_Cout1_1 = 0
        Int_Cout1_2 = 0
        Int_Cout1_3 = 0
        Int_Cout1_4 = 0
        C_PUMP_grad_1 = C_PUMP_grad_I1
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        x0_1[4*z_size + 1]   =  Int_Cout1_1
        x0_1[5*z_size + 1]   =  Int_Cout1_2
        x0_1[6*z_size + 1]   =  Int_Cout1_3
        x0_1[7*z_size + 1]   =  Int_Cout1_4
        # print(x0_1)
        outputs_1 = self.simulation(x0_1, {'Q':Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_I1})
        t = outputs_1['r']['Time'].to_numpy()
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 2 
        # Takes in the breakthrough of column 1, cin changes in time for column 2
        # Interconnected columns
        Qin_2 = Q_col_2
        QPUMP2_I1 = Qin_2 - Q_I1
        C_PUMP2_1 = Cmod2
        C_PUMP2_2 = 0
        C_PUMP2_3 = 0
        C_PUMP2_4 = 0
        Int_Cout2_1 = 0
        Int_Cout2_2 = 0
        Int_Cout2_3 = 0
        Int_Cout2_4 = 0
        Cin2_1 = Qin_1*C1out_1/Qin_2 + QPUMP2_I1*C_PUMP2_1/Qin_2
        Cin2_2 = Qin_1*C1out_2/Qin_2 + QPUMP2_I1*C_PUMP2_2/Qin_2
        Cin2_3 = Qin_1*C1out_3/Qin_2 + QPUMP2_I1*C_PUMP2_3/Qin_2
        Cin2_4 = Qin_1*C1out_4/Qin_2 + QPUMP2_I1*C_PUMP2_4/Qin_2
        x0_2[0]   =  Cin2_1[0]
        x0_2[z_size]   =  Cin2_2[0]
        x0_2[2*z_size]   =  Cin2_3[0]
        x0_2[3*z_size]   =  Cin2_4[0]
        x0_2[4*z_size + 1]   =  Int_Cout2_1
        x0_2[5*z_size + 1]   =  Int_Cout2_2
        x0_2[6*z_size + 1]   =  Int_Cout2_3
        x0_2[7*z_size + 1]   =  Int_Cout2_4
        outputs_2 = self.simulation_interconnected(x0_2, t, Cin2_1, Cin2_2, Cin2_3, Cin2_4, {'Q':Qin_2})
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'I1S1'
        outputs_2['r']['step'] = 'I1S1'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step B1S1 ---------------------------------- #
        # Batch (right column is column 1)
        last_t    = outputs_1['last_t']
        # Simulate Column 1
        Qin_1 = Q_B
        Cin1_1 = x0_1[0]
        Cin1_2 = 0
        Cin1_3 = 0
        Cin1_4 = 0
        C_PUMP_grad_1 = C_PUMP_grad_B1
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        outputs_1 = self.simulation(x0_1, {'Q':Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_B1})
        t = outputs_1['r']['Time'].to_numpy()
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 2 
        # Takes in the  fresh feed
        # Batch columns, 
        Qin_2 = Q_col_2
        Cin2_1 = CFEED1
        Cin2_2 = CFEED2
        Cin2_3 = CFEED3
        Cin2_4 = CFEED4
        C_PUMP_grad_2 = 0
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        outputs_2 = self.simulation(x0_2, {'Q':Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_B1})
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'B1S1'
        outputs_2['r']['step'] = 'B1S1'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step I2S1 ---------------------------------- #
        # Interconnected  (right column is column 1)
        last_t    = outputs_1['last_t']
        # Simulate Column 1
        Qin_1 = Q_I2
        Cin1_1 = x0_1[0]
        Cin1_2 = 0
        Cin1_3 = 0
        Cin1_4 = 0
        C_PUMP_grad_1 = C_PUMP_grad_I2
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        outputs_1 = self.simulation(x0_1, {'Q':Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_I2})
        t = outputs_1['r']['Time'].to_numpy()
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 2 
        # Takes in the breakthrough of column 1, cin changes in time for column 2
        # Interconnected columns
        Qin_2 = Q_col_2
        QPUMP2_I2 = Qin_2 - Q_I2
        C_PUMP2_1 = Cmod2
        C_PUMP2_2 = 0
        C_PUMP2_3 = 0
        C_PUMP2_4 = 0
        Cin2_1 = Qin_1*C1out_1/Qin_2 + QPUMP2_I2*C_PUMP2_1/Qin_2
        Cin2_2 = Qin_1*C1out_2/Qin_2 + QPUMP2_I2*C_PUMP2_2/Qin_2
        Cin2_3 = Qin_1*C1out_3/Qin_2 + QPUMP2_I2*C_PUMP2_3/Qin_2
        Cin2_4 = Qin_1*C1out_4/Qin_2 + QPUMP2_I2*C_PUMP2_4/Qin_2
        x0_2[0]   =  Cin2_1[0]
        x0_2[z_size]   =  Cin2_2[0]
        x0_2[2*z_size]   =  Cin2_3[0]
        x0_2[3*z_size]   =  Cin2_4[0]
        outputs_2 = self.simulation_interconnected(x0_2, t, Cin2_1, Cin2_2, Cin2_3, Cin2_4, {'Q':Qin_2})
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'I2S1'
        outputs_2['r']['step'] = 'I2S1'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step B2S1 ---------------------------------- #
        # Batch (right column is column 1)
        last_t    = outputs_1['last_t']
        # Simulate Column 1
        Qin_1 = Q_B
        Cin1_1 = Cmod1 - (C_PUMP_grad_B2 + C_PUMP_grad_B2_col_2) * T_B2
        Cin1_2 = 0
        Cin1_3 = 0
        Cin1_4 = 0
        C_PUMP_grad_1 = C_PUMP_grad_B2
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        outputs_1 = self.simulation(x0_1, {'Q':Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_B2})
        t = outputs_1['r']['Time'].to_numpy()
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 2 
        # Takes in the  fresh feed
        # Batch columns, 
        Qin_2 = Q_col_2
        Cin2_1 = Cmod1 - C_PUMP_grad_B2_col_2 * T_B2
        Cin2_2 = 0
        Cin2_3 = 0
        Cin2_4 = 0
        C_PUMP_grad_2 = C_PUMP_grad_B2_col_2
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        outputs_2 = self.simulation(x0_2, {'Q':Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_B2})
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'B2S1'
        outputs_2['r']['step'] = 'B2S1'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])


        # ---------------------------- Half cycle finished ---------------------------- #

        # ---------------------------------- Step I1S2 ---------------------------------- #
        # Interconnected  (right column is column 2)
        last_t    = outputs_2['last_t']
        # Simulate Column 2
        Qin_2 = Q_I1
        Cin2_1 = Cmod1
        Cin2_2 = 0
        Cin2_3 = 0
        Cin2_4 = 0
        Int_Cout2_1 = 0
        Int_Cout2_2 = 0
        Int_Cout2_3 = 0
        Int_Cout2_4 = 0
        C_PUMP_grad_2 = C_PUMP_grad_I1
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        x0_2[4*z_size + 1]   =  Int_Cout2_1
        x0_2[5*z_size + 1]   =  Int_Cout2_2
        x0_2[6*z_size + 1]   =  Int_Cout2_3
        x0_2[7*z_size + 1]   =  Int_Cout2_4
        outputs_2 = self.simulation(x0_2, {'Q':Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_I1})
        t = outputs_2['r']['Time'].to_numpy()
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 1 
        # Takes in the breakthrough of column 2, cin changes in time for column 1
        # Interconnected columns
        Qin_1 = Q_col_2
        QPUMP2_I1 = Qin_1 - Q_I1
        C_PUMP2_1 = Cmod2
        C_PUMP2_2 = 0
        C_PUMP2_3 = 0
        C_PUMP2_4 = 0
        Int_Cout1_1 = 0
        Int_Cout1_2 = 0
        Int_Cout1_3 = 0
        Int_Cout1_4 = 0
        Cin1_1 = Qin_2*C2out_1/Qin_1 + QPUMP2_I1*C_PUMP2_1/Qin_1
        Cin1_2 = Qin_2*C2out_2/Qin_1 + QPUMP2_I1*C_PUMP2_2/Qin_1
        Cin1_3 = Qin_2*C2out_3/Qin_1 + QPUMP2_I1*C_PUMP2_3/Qin_1
        Cin1_4 = Qin_2*C2out_4/Qin_1 + QPUMP2_I1*C_PUMP2_4/Qin_1
        x0_1[0]   =  Cin1_1[0]
        x0_1[z_size]   =  Cin1_2[0]
        x0_1[2*z_size]   =  Cin1_3[0]
        x0_1[3*z_size]   =  Cin1_4[0]
        x0_1[4*z_size + 1]   =  Int_Cout1_1
        x0_1[5*z_size + 1]   =  Int_Cout1_2
        x0_1[6*z_size + 1]   =  Int_Cout1_3
        x0_1[7*z_size + 1]   =  Int_Cout1_4
        outputs_1 = self.simulation_interconnected(x0_1, t, Cin1_1, Cin1_2, Cin1_3, Cin1_4, {'Q':Qin_1})
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'I1S2'
        outputs_2['r']['step'] = 'I1S2'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step B1S2 ---------------------------------- #
        # Batch (right column is column 2)
        last_t    = outputs_2['last_t']
        # Simulate Column 2
        Qin_2 = Q_B
        Cin2_1 = x0_2[0]
        Cin2_2 = 0
        Cin2_3 = 0
        Cin2_4 = 0
        C_PUMP_grad_2 = C_PUMP_grad_B1
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        outputs_2 = self.simulation(x0_2, {'Q':Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_B1})
        t = outputs_2['r']['Time'].to_numpy()
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 1 
        # Takes in the  fresh feed
        # Batch columns, 
        Qin_1 = Q_col_2
        Cin1_1 = CFEED1
        Cin1_2 = CFEED2
        Cin1_3 = CFEED3
        Cin1_4 = CFEED4
        C_PUMP_grad_1 = 0
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        outputs_1 = self.simulation(x0_1, {'Q':Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_B1})
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'B1S2'
        outputs_2['r']['step'] = 'B1S2'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step I2S2 ---------------------------------- #
        # Interconnected  (right column is column 2)
        last_t    = outputs_2['last_t']
        # Simulate Column 2
        Qin_2 = Q_I2
        Cin2_1 = x0_2[0]
        Cin2_2 = 0
        Cin2_3 = 0
        Cin2_4 = 0
        C_PUMP_grad_2 = C_PUMP_grad_I2
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        outputs_2 = self.simulation(x0_2, {'Q':Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_I2})
        t = outputs_2['r']['Time'].to_numpy()
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 1
        # Takes in the breakthrough of column 2, cin changes in time for column 1
        # Interconnected columns
        Qin_1 = Q_col_2
        QPUMP2_I2 = Qin_1 - Q_I2
        C_PUMP2_1 = Cmod2
        C_PUMP2_2 = 0
        C_PUMP2_3 = 0
        C_PUMP2_4 = 0
        Cin1_1 = Qin_2*C2out_1/Qin_1 + QPUMP2_I2*C_PUMP2_1/Qin_1
        Cin1_2 = Qin_2*C2out_2/Qin_1 + QPUMP2_I2*C_PUMP2_2/Qin_1
        Cin1_3 = Qin_2*C2out_3/Qin_1 + QPUMP2_I2*C_PUMP2_3/Qin_1
        Cin1_4 = Qin_2*C2out_4/Qin_1 + QPUMP2_I2*C_PUMP2_4/Qin_1
        x0_1[0]   =  Cin1_1[0]
        x0_1[z_size]   =  Cin1_2[0]
        x0_1[2*z_size]   =  Cin1_3[0]
        x0_1[3*z_size]   =  Cin1_4[0]
        outputs_1 = self.simulation_interconnected(x0_1, t, Cin1_1, Cin1_2, Cin1_3, Cin1_4, {'Q':Qin_1})
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'I2S2'
        outputs_2['r']['step'] = 'I2S2'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])

        # ---------------------------------- Step B2S2 ---------------------------------- #
        # Batch (right column is column 2)
        last_t    = outputs_2['last_t']
        # Simulate Column 2
        Qin_2 = Q_B
        Cin2_1 =  Cmod1 - (C_PUMP_grad_B2 + C_PUMP_grad_B2_col_2) * T_B2
        Cin2_2 = 0
        Cin2_3 = 0
        Cin2_4 = 0
        C_PUMP_grad_2 = C_PUMP_grad_B2
        x0_2[0]   =  Cin2_1
        x0_2[z_size]   =  Cin2_2
        x0_2[2*z_size]   =  Cin2_3
        x0_2[3*z_size]   =  Cin2_4
        outputs_2 = self.simulation(x0_2, {'Q': Qin_2, 'C_PUMP_grad': C_PUMP_grad_2, 't0': last_t, 'tF': last_t + T_B2})
        t = outputs_2['r']['Time'].to_numpy()
        x0_2      = outputs_2['last_state']
        C2out_1    = outputs_2['r'][f'C1_{1}'].to_numpy()
        C2out_2    = outputs_2['r'][f'C2_{1}'].to_numpy()
        C2out_3    = outputs_2['r'][f'C3_{1}'].to_numpy()
        C2out_4    = outputs_2['r'][f'C4_{1}'].to_numpy()

        # Simulate Column 1 
        # Takes in the  fresh feed
        # Batch columns, 
        Qin_1 = Q_col_2
        Cin1_1 = Cmod1 - C_PUMP_grad_B2_col_2 * T_B2
        Cin1_2 = 0
        Cin1_3 = 0
        Cin1_4 = 0
        C_PUMP_grad_1 = C_PUMP_grad_B2_col_2
        x0_1[0]   =  Cin1_1
        x0_1[z_size]   =  Cin1_2
        x0_1[2*z_size]   =  Cin1_3
        x0_1[3*z_size]   =  Cin1_4
        outputs_1 = self.simulation(x0_1, {'Q': Qin_1, 'C_PUMP_grad': C_PUMP_grad_1, 't0': last_t, 'tF': last_t + T_B2})
        x0_1      = outputs_1['last_state']
        C1out_1    = outputs_1['r'][f'C1_{1}'].to_numpy()
        C1out_2    = outputs_1['r'][f'C2_{1}'].to_numpy()
        C1out_3    = outputs_1['r'][f'C3_{1}'].to_numpy()
        C1out_4    = outputs_1['r'][f'C4_{1}'].to_numpy()

        # Record step for outputs
        outputs_1['r']['step'] = 'B2S2'
        outputs_2['r']['step'] = 'B2S2'
        o1.append(outputs_1['r'])
        o2.append(outputs_2['r'])


        # ---------------------------- Cycle finished ---------------------------- #


        # ----- Pack up outputs ----- #
        # Concatenate data from all simulations
        df1 = pd.concat(o1, ignore_index = True).drop_duplicates('Time', keep = 'last')
        df2 = pd.concat(o2, ignore_index = True).drop_duplicates('Time', keep = 'last')

                # ---------------------------- Adjust Mass Balances ---------------------------- #
        # Column 1

        # Assign flowrates
        flow_rate_values_1 = [Q_I1, Q_B, Q_I2, Q_B, Q_col_2]                   # Flow rate values for column 1
        step_counts_1 = [T_I1, T_B1, T_I2, T_B2, T_I1+T_B1+T_I2+T_B2]          # Steps for each flow rate for column 1
        flow_rate_values_2 = [Q_col_2, Q_B, Q_I1, Q_B, Q_I2]                   # Flow rate values for column 2
        step_counts_2 = [T_I1+T_B1+T_I2+T_B2, T_I1, T_B1, T_I2, T_B2]          # Steps for each flow rate for column 2

        # Use numpy.repeat to create the flow rate array
        Q1 = np.concatenate([np.repeat(value, count) for value, count in zip(flow_rate_values_1, step_counts_1)])        
        Q2 = np.concatenate([np.repeat(value, count) for value, count in zip(flow_rate_values_2, step_counts_2)])

        # Calculate outlet mass 
        mass_in_2 = df1['m_2_0']
        mass_out_2 = df1['m_2_1']
        mass_in_3 = df1['m_3_0']
        mass_out_3 = df1['m_3_1']
        mass_in_4 = df1['m_4_0']
        mass_out_4 = df1['m_4_1']

        # Calculate the mass difference
        mass_diff_2 = mass_in_2 - mass_out_2
        mass_diff_3 = mass_in_3 - mass_out_3
        mass_diff_4 = mass_in_4 - mass_out_4

        # Normilise masses
        norm_2 = df1['m_2_1']/sum(df1['m_2_1'])
        norm_3 = df1['m_3_1']/sum(df1['m_3_1'])
        norm_4 = df1['m_4_1']/sum(df1['m_4_1'])

        # Recalculate the output masses
        df1['m_2_1'] += norm_2*sum(mass_diff_2)
        df1['m_3_1'] += norm_3*sum(mass_diff_3)
        df1['m_4_1'] += norm_4*sum(mass_diff_4)

        # Adjust the elution profile
        df1['C2_1'] = df1['m_2_1']/Q1
        df1['C3_1'] = df1['m_3_1']/Q1
        df1['C4_1'] = df1['m_4_1']/Q1

        # Column 2

        # Calculate outlet mass 
        mass_in_2 = df2['m_2_0']
        mass_out_2 = df2['m_2_1']
        mass_in_3 = df2['m_3_0']
        mass_out_3 = df2['m_3_1']
        mass_in_4 = df2['m_4_0']
        mass_out_4 = df2['m_4_1']

        # Calculate the mass difference
        mass_diff_2 = mass_in_2 - mass_out_2
        mass_diff_3 = mass_in_3 - mass_out_3
        mass_diff_4 = mass_in_4 - mass_out_4

        # Normilise masses
        norm_2 = df2['m_2_1']/sum(df2['m_2_1'])
        norm_3 = df2['m_3_1']/sum(df2['m_3_1'])
        norm_4 = df2['m_4_1']/sum(df2['m_4_1'])

        # Recalculate the output masses
        df2['m_2_1'] += norm_2*sum(mass_diff_2)
        df2['m_3_1'] += norm_3*sum(mass_diff_3)
        df2['m_4_1'] += norm_4*sum(mass_diff_4)

        # Adjust the elution profile
        df2['C2_1'] = df2['m_2_1']/Q2
        df2['C3_1'] = df2['m_3_1']/Q2
        df2['C4_1'] = df2['m_4_1']/Q2

        # Exlude some headers from the data (variables which are shared across columns)
        params_l = list(params.keys()).copy()
        params_l.remove('t0')
        params_l.remove('tF')
        # params_l.remove('Q')
        params_l.remove('C_PUMP_grad')
        # shared_labels = ['Time', 'status', 'step'] + params_l
        shared_labels = ['Time', 'step'] + params_l
        df = df1.loc[:, df1.columns.isin(shared_labels)]
        df1 = df1.loc[:, ~df1.columns.isin(shared_labels)]
        df2 = df2.loc[:, ~df2.columns.isin(shared_labels)]
        # Rename the keys of the dataframe to reflect the column number
        new_keys_1 = {i: 'COL1_' + i for i in list(df1.keys()[df1.keys() != 'Time'])}
        df1 = df1.rename(columns = new_keys_1)
        new_keys_2 = {i: 'COL2_' + i for i in list(df2.keys()[df2.keys() != 'Time'])}
        df2 = df2.rename(columns = new_keys_2)
        # Concatenate all data
        df = pd.concat([df, df1, df2], axis = 1)
        df = pd.concat([df, df1], axis = 1)
        df = df.reset_index(drop = True)

        outputs = {}
        outputs['r'] = df
        outputs['COL1_last_state']  = x0_1
        outputs['COL2_last_state']  = x0_2
        outputs['last_t'] = outputs_1['last_t']
        return outputs

    def simulate_process(self,  n_cycles, params = 'def'):
        """
        Simulate the process for n_cycles number of cycles
        """
        
        # Update parameters
        if params != 'def':
            self.params.update(params)
        # Unpack params
        params = self.params.copy()

        # Output container
        output_list = []

        # First cycle parameters
        last_t = 0.0
        last_state = None

        # Loop
        for i in range(n_cycles):
            output = self.one_cycle(last_t, last_state, params)
            last_state = [output['COL1_last_state'], output['COL2_last_state']]
            last_t = output['last_t']

            # Record cycle_no
            output['r']['cycle_no'] = i + 1
            output_list.append(output['r'])
        df = (pd.concat(output_list, ignore_index = True).drop_duplicates('Time', \
            keep = 'last')).reset_index(drop = True)

        return df

# %%
params = {}
initial_params = {
    # Concentrations of feed stream introduced in column executing recycling tasks
    # 1: Modifier, 2: Weak Impurities, 3: Product, 4: Strong Impurities
    'CFEED1': val,
    'CFEED2': val,
    'CFEED3': val,
    'CFEED4': val,

    'Cmod1': val, # Initial modifier concentration of column executing elution tasks
    'Cmod2': val, # Initial modifier concentration of column executing recycling tasks

    # Modifier gradient of different process steps for column 1
    'C_PUMP_grad': val, 
    'C_PUMP_grad_I1': val, 
    'C_PUMP_grad_B1': val, 
    'C_PUMP_grad_I2': val, 
    'C_PUMP_grad_B2': val, 
    'C_PUMP_grad_B2_col_2': val, # Modifier gradient of column executing recycling tasks

    # Flowrate values of column executing elution tasks for different process steps
    'Q': val,   
    'Q_I1': val, 
    'Q_B': val, 
    'Q_I2': val,
    'Q_col_2': val,  # Flowrate of column executing recycling tasks

    # Step durations
    'T_I1': val, 
    'T_B1': val, 
    'T_I2': val, 
    'T_B2': val, 

    # Parameter values - First number: Column, Second number = Species
    'A1_1': val,
    'A1_2': val,
    'A1_3': val,
    'A1_4': val,
    'A2_1': val,
    'A2_2': val,
    'A2_3': val,
    'A2_4': val,
    'A3_1': val,
    'A3_2': val,
    'A3_3': val,
    'A3_4': val,
    'A4_1': val,
    'A4_2': val,
    'A4_3': val,
    'A4_4': val,
    'A5_1': val,
    'A5_2': val,
    'A5_3': val,
    'A5_4': val,
    'A6_1': val,
    'A6_2': val,
    'A6_3': val,
    'A6_4': val,
    'A7_1': val,
    'A7_2': val,
    'A7_3': val,
    'A7_4': val,
    'A8_1': val,
    'A8_2': val,
    'A8_3': val,
    'A8_4': val,

    'HMOD': val,

    # Hyperparameters for simulation of column
    't0': 0,   # Initial time of column simulation
    'tF': val,  # Final time of column simulation - time of one process cycle
    'dt': 1,   # Reporting time step
    'dz': 1    # z-space discretisation step (normalised: 0 - 1)
}

# %%
# Load ANN model
Cout_ANN  = load_model('Model_Name.pkl')

# %%
params.update(initial_params)
m = MCSGP(params)
df = m.simulate_process(1)


