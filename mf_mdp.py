import numpy as np
from scipy import linalg
from sklearn.preprocessing import normalize

#This function returns matrices p_0, p_\uparrow and p_\downarrow.
#n is total number of states (temperatures) for ON state of TCL.
#q is number of states (temperatures) in uncomfortable zone for ON state of TCL.
#epsilon is diffusion rate (probability of self loop and double step)

def build_transition_matrix(n, q, epsilon):
    
    p_up = np.eye(n+1)[:-1,1:]*(1-2*epsilon) + \
    epsilon*np.eye(n)+epsilon*np.eye(n+2)[:-2,2:]
    
    p_up[-1, -1] = 0
    p_up[-1, -2] = 1 - epsilon
    
    p_down = np.eye(n+1)[1:,:-1]*(1-2*epsilon) + \
    epsilon*np.eye(n)+epsilon*np.eye(n+2)[2:,:-2]
    
    p_down[0, 0] = 0
    p_down[0, 1] = 1 - epsilon
    
    p0 = np.kron(np.array([[1.,0.], [0.,0.]]), p_down) + \
    np.kron(np.array([[0.,0.],[0.,1.]]), p_up)
    
    p0[n-1, 2*n-1] = 1
    p0[n, 0] = 1
    p_ru = np.eye(n)
    p_ru[:-q, :-q] = 0
    p_ru[-1, -1] = 0
    p_ld = np.eye(n)
    p_ld[q:, q:] = 0
    p_ld[0, 0] = 0
    p_lu = np.eye(n+1)[1:,:-1]
    p_lu[q-1:, q-1:] = 0
    p_rd = np.eye(n+1)[:-1,1:]
    p_rd[:1-q, :1-q] = 0
    
    p_down_to_up = np.kron([[0.,1.],[0.,0.]], p_ru) - \
    np.kron([[0.,0.],[0.,1.]], p_rd)
    
    p_up_to_down = np.kron([[0.,0.],[1.,0.]], p_ld) - \
    np.kron([[1.,0.],[0.,0.]], p_lu)
    
    return p0, p_down_to_up, p_up_to_down


#This function returns consumption vector
#n is total number of states (temperatures) for ON state of conditioner.

def consumption_vecor(n = 20):
    return np.kron([1., 0.], np.ones(n))


#This is boundary function which ensure stochasticity of total transition matrix

def f(x, epsilon):
    if x<1. - 2*epsilon:
        return x
    else:
        return 1. - 2*epsilon
    

#This class contains object(mf_mdp_model) with all parameters of discrete TCL,
#such as initial state (now is fixed),
#number of states in uncomfortable zone for ON positionv(q), 
#total number of states in ON position (n), Poisson switchings rate (r), 
#rate of mean field control (alpha).


class mf_mdp_model():
    
    #initializing of the object. Matrices p_0, p_down, 
    #p_up are produced via fucntions given above.
    
    #n is total numer of states in ON position
    #q is number of states in uncomfortable zone for ON position
    #epsilon is diffusion rate
    #alpha is strength of non-linearity
    #r is poisson switchings rate
    
    def __init__(self, n=20, q=6, epsilon=0.05, alpha=1., r=0.1):
        
        P = np.zeros(2*n)
        P[n//4] = 1.
        self.initial_state = P
        self.p = build_transition_matrix(n, q, epsilon)
        self.alpha = alpha
        self.r = r
        self.epsilon = epsilon
        self.U = consumption_vecor(n)
        
    #method direct_simulation finds solution of Master equation straightforwardly
    
    #time_steps is total number of time steps in simulation
    
    #method returns consumption vector and h1
    #(l1 distance between steady state and current state)
        
    def direct_simulation(self, time_steps=600):
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        alpha = self.alpha
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
    
        P = self.initial_state
        consumption = np.array([])
        h1 = np.array([])
        for i in range(time_steps):
            
            transition = p_0 + p_up*f(r*((2*(1 - np.dot(consumption_vector, \
            P)))**alpha), self.epsilon) + \
            p_down*f(r*((2*np.dot(consumption_vector, P))**alpha),\
            self.epsilon)
            
            P = np.dot(transition, P)
            consumption = np.append(consumption, \
            np.absolute(2*np.dot(consumption_vector, P) - 1))
            
            h1 = np.append(h1, np.linalg.norm(Q-P ,ord = 1))
            
        return consumption, h1
    
    #method direct_simulation_dens finds solution of 
    #Master equation straightforwardly
    
    #time_steps is total number of time steps in simulation
    
    #method returns array of PDF for each time, consumption vector and h1 
    #(l1 distance between steady state and current state)
    
    def direct_simulation_dens(self, time_steps=600):
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        alpha = self.alpha
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
    
        P = self.initial_state
        consumption = np.array([])
        h1 = np.array([])
        sol = np.array([])
        for i in range(time_steps):
            transition = p_0 + p_up*f(r*((2*(1 - \
            np.dot(consumption_vector, P)))**alpha), self.epsilon) + \
            p_down*f(r*((2*np.dot(consumption_vector, P))**alpha), self.epsilon)
            
            P = np.dot(transition, P)
            sol = np.append(sol, P)
            
            consumption = np.append(consumption, np.absolute(2*np.dot(\
            consumption_vector, P) - 1))
            
            h1 = np.append(h1, np.linalg.norm(Q-P ,ord = 1))
        sol = sol.reshape(time_steps, -1)
        
        return sol, h1, consumption
    
    #method gap returns gap between minimal relaxation rate of consumption and
    #minimal relaxation rate of whole ensemble
    
    #give_lambda is binary parameter which can be either True or False
    
    #if this parameter is True this method returns gap, min relax. constant for
    #consumption and whole ensemble
    #if this parameter is False, method returns only gap
    
    def gap(self, give_lambda=False):
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        alpha = self.alpha
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        L = (p_0.T).dot(p_0)
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
        
        SS = p_0 + r*p_up + r*p_down + \
        2*r*alpha*np.einsum("i,j->ij", np.dot(p_down, Q), consumption_vector) \
        - 2*r*alpha*np.einsum("i,j->ij", np.dot(p_up, Q), consumption_vector)
        
        lin_ev, _, r_vec = linalg.eig(SS, left = True)
        
        order = (-np.log(np.abs(lin_ev))).argsort()
        
        lin_ev = lin_ev[order][1:]
        r_vec = r_vec[:, order][:, 1:]
        
        ind = np.abs(consumption_vector.dot(r_vec))>1e-10
        im_part = np.angle(lin_ev)
        re_part = -np.log(np.abs(lin_ev))
        relax_H = re_part#[np.logical_not(ind)]
        relax_U = re_part[ind]
        
        gap = np.min(relax_U) - np.min(relax_H)
        
        if give_lambda == True:
            return gap, np.min(relax_U), np.min(relax_H)
        else:
            return gap
    
    #method relax_constants returns set of all relaxation constants
    #for linearized transition matrix
    
    def relax_constants(self):
        
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        alpha = self.alpha
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        L = (p_0.T).dot(p_0)
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
    
        SS = p_0 + r*p_up + r*p_down + 2*r*alpha*np.einsum("i,j->ij",\
        np.dot(p_down, Q), consumption_vector) - \
        2*r*alpha*np.einsum("i,j->ij", np.dot(p_up, Q), consumption_vector)
    
        lin_ev, _, r_vec = linalg.eig(SS, left = True)
        
        lin_ev = lin_ev[(np.einsum('ji,jk,ki->i', r_vec.conj(),\
        -L, r_vec)/(np.einsum('ji,ji->i', r_vec.conj(), r_vec)) + \
        1e-10*np.angle(lin_ev)).argsort()]
    
        im_part = np.angle(lin_ev)
        re_part = -np.log(np.abs(lin_ev))
        
        return re_part, im_part
    
    #relax_constant_tracking gives us an evolution of relaxtion constants with
    #alpha. This function preservs order of 
    #relaxation contants with evolution
    
    #alpha_max is maximum value of alpha
    #d_alpha is discretization (step)
    
    #method returns array of real parts of eigenvalues, 
    #array of imaginary parts of eigenvalues and set of corresponding alphas
    
    def relax_constant_tracking(self, alpha_max, d_alpha):
        
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
        check_vec = right_v[:, np.abs(np.angle(ev) + 1e-5).argsort()]
        
        set_of_constants = np.array([])
        
        for alpha in np.arange(0, alpha_max, d_alpha):
            
            SS = p_0 + r*p_up + r*p_down + 2*r*alpha*np.einsum("i,j->ij",\
            np.dot(p_down, Q), consumption_vector) - \
            2*r*alpha*np.einsum("i,j->ij", np.dot(p_up, Q), consumption_vector)
            
            lin_ev, l_vec, r_vec = linalg.eig(SS, left = True)
            order = np.abs(np.conj(l_vec.T).dot(check_vec)).argmax(0)
            check_vec = r_vec[:, order]
            set_of_constants = np.append(set_of_constants, lin_ev[order])
        set_of_constants = set_of_constants.reshape(int(alpha_max/d_alpha), -1)
        im_part = np.angle(set_of_constants*np.exp(1j*1e-4))
        re_part = -np.log(np.abs(set_of_constants))
            
        return re_part, im_part, np.arange(0, alpha_max, d_alpha)
    
    #method steady_state returns steady state of the system
    
    def steady_state(self):
        
        p_0, p_up, p_down = self.p
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
        
        return Q
    
    #eid_sys method returns eigensystem
        
    def eig_sys(self):
        
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        alpha = self.alpha
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
    
        SS = p_0 + r*p_up + r*p_down + 2*r*alpha*np.einsum("i,j->ij", np.dot(p_down, Q), consumption_vector) - 2*r*alpha*np.einsum("i,j->ij", np.dot(p_up, Q), consumption_vector)
        
        return linalg.eig(SS, left = True)
    
    #method eig_sys_tracking tracks for the eigensystem with changing of alpha
    
    #alpha_max is maximum value of alpha
    #d_alpha is discretization (step)
    
    #method returns set of eigenvalues, set of right eig. vectors, set of left
    #eig. vectors and corresponding alphas
    
    def eig_sys_tracking(self, alpha_max, d_alpha):
        
        p_0, p_up, p_down = self.p
        consumption_vector = self.U
        r = self.r
        
        S = p_0 + r*p_down + r*p_up
        ev, left_v, right_v = linalg.eig(S, left = True)
        steady = right_v[:,0]
        Q = steady/steady.sum()
        check_vec = right_v[:, np.abs(np.angle(ev) + 1e-5).argsort()]
        
        set_of_ev = np.expand_dims(ev, axis=0)
        set_of_lv = np.expand_dims(left_v, axis=0)
        set_of_rv = np.expand_dims(right_v, axis=0)
        
        for alpha in np.arange(0, alpha_max, d_alpha):
            
            SS = p_0 + r*p_up + r*p_down + 2*r*alpha*np.einsum("i,j->ij", np.dot(p_down, Q), consumption_vector) - 2*r*alpha*np.einsum("i,j->ij", np.dot(p_up, Q), consumption_vector)
            lin_ev, l_vec, r_vec = linalg.eig(SS, left = True)
            order = np.abs(np.conj(l_vec.T).dot(check_vec)).argmax(0)
            check_vec = r_vec[:, order]
            set_of_ev = np.append(set_of_ev, np.expand_dims(lin_ev[order], axis=0), axis=0)
            set_of_lv = np.append(set_of_lv, np.expand_dims(l_vec[:,order], axis=0), axis=0)
            set_of_rv = np.append(set_of_rv, np.expand_dims(r_vec[:,order], axis=0), axis=0)
            
        return set_of_ev, set_of_lv, set_of_rv, np.arange(0, alpha_max, d_alpha)