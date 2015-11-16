import sympy
import re
import copy
import numpy
import itertools
import matplotlib.pylab as plt
import pysb
import stoichiometry_analysis as sto
import math
from pysb.integrate import odesolve
from collections import OrderedDict

# def _parse_name(spec):
#     m = spec.monomer_patterns
#     lis_m = []
#     for i in range(len(m)):
#         tmp_1 = str(m[i]).partition('(')
#         tmp_2 = re.findall(r"(?<=\').+(?=\')",str(m[i]))
#         if tmp_2 == []: lis_m.append(tmp_1[0])
#         else:
#             lis_m.append(''.join([tmp_1[0],tmp_2[0]]))
#     return '_'.join(lis_m)

def _Heaviside_num(x):
    return 0.5*(numpy.sign(x)+1)

class Tropical:
    def __init__(self, model):
        self.model              = model
        self.tspan              = None
        self.y                  = None  # ode solution, numpy array
        self.param_values       = None
        self.passengers         = None
        self.sto_conserved      = None
        self.conservation       = None
        self.conserve_var       = None
        self.value_conservation = {}
        self.tro_species        = {}
        self.driver_signatures  = None
        self.passenger_signatures = None
        self.mon_names          = {}

    def __repr__(self):
        return "<%s '%s' (passengers: %s, cycles: %d) at 0x%x>" % \
            (self.__class__.__name__, self.model.name,
             self.passengers.__repr__(),
             len(self.cycles),
             id(self))

    def tropicalize(self,tspan=None, param_values=None, ignore=1, epsilon=1, rho=1, verbose=True):
        
        if verbose: print "Solving Simulation"
        
        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")
        
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

#         subs = dict((p, param_values[i]) for i, p in enumerate(self.model.parameters))
        new_pars = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))
        self.param_values = new_pars
              
        self.y = odesolve(self.model, self.tspan, self.param_values) 
          
        if verbose: print "Getting Passenger species"
        self.find_passengers(self.y[ignore:], verbose, epsilon)
        if verbose: print "Computing conservation relations"
        self.sto_conserved = sto.conservation_relations(self.model)
        if verbose: print "Computing Conservation laws"
        (self.conservation, self.conserve_var, self.value_conservation) = self.mass_conserved(self.sto_conserved, self.y[ignore:])
        if verbose: print "Pruning Equations"
        self.pruned = self.pruned_equations(self.y[ignore:], rho)
        if verbose: print "Solving pruned equations"
        self.sol_pruned = self.solve_pruned()
        if verbose: print "equation to tropicalize"
        self.eqs_for_tropicalization = self.equations_to_tropicalize()
        if verbose: print "Getting tropicalized equations"
        self.tropical_eqs = self.final_tropicalization()
        self.data_drivers(self.y[ignore:])
#         
        return 

    def find_passengers(self, y, verbose=False, epsilon=None, ptge_similar=0.9, plot=False):
        sp_imposed_trace = {}
        self.passengers = []
        solved_pol = []                #  list of solved polynomial equations
        diff_eqs = []                  #  list of differential equations   

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            eq        = eq.subs('__s%d' % i, '__s%dstar' % i)
            sol       = sympy.solve(eq, sympy.Symbol('__s%dstar' % i))        # Find equation of imposed trace
            sp_imposed_trace[i] = sol
       
        for k in sp_imposed_trace.keys():
            distance_imposed = 999
            for idx, solu in enumerate(sp_imposed_trace[k]):  
                for p in self.param_values: solu = solu.subs(p, self.param_values[p])            
                args = []                                                         #arguments to put in the lambdify function
                variables = [atom for atom in solu.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                f = sympy.lambdify(variables, solu, modules = dict(sqrt=numpy.lib.scimath.sqrt) )
                for l in variables:
                    args.append(y[:][str(l)])
                if any(isinstance(n,complex) for n in f(*args)): 
                    print 'solution'+ ' ' + '%d'%idx + ' ' + 'from equation' + ' ' + str(k) + ' ' + 'is complex'
                    continue
                elif any(n<0 for n in f(*args)):
                    print 'solution'+ ' ' + '%d'%idx + ' ' + 'from equation' + ' ' + str(k) + ' ' + 'is negative'
                    continue                  
                hey = abs(numpy.log10(f(*args)) - numpy.log10(y['__s%d'%k]))
                if max(hey) < distance_imposed: distance_imposed = max(hey)
                if plot:
                    plt.figure()
                    plt.semilogy(self.tspan[1:],f(*args), 'r--', linewidth=5, label= 'imposed')
                    plt.semilogy(self.tspan[1:],y['__s%d'%k], label='full')
                    plt.legend(loc=0)
                    plt.xlabel('time')
                    plt.ylabel('population')
                    if max(hey) < epsilon :
                        plt.title(str(self.model.species[k])+'passenger', fontsize=20)    
                    else: plt.title(self.model.species[k], fontsize=20)   
                
            if distance_imposed < epsilon : 
                self.passengers.append(k)
                


                
#             s_points = sum(w < epsilon for w in hey)
#             if s_points > ptge_similar*len(hey) : 
#                 self.passengers.append(diff_eqs[k])
        plt.show()
        return self.passengers

    #This function finds conservation laws from the conserved cycles
    def mass_conserved(self, conservation_laws, y, verbose=False):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(self.model)
        h = [] # Array to hold conservation equation
        g = [] # Array to hold corresponding lists of free variables in conservation equations
        value_constants = {} #Dictionary that storage the value of each constant
        for i, item in enumerate(conservation_laws):
            b = 0
            u = 0
            for j, specie in enumerate(item):
                b += self.model.odes[conservation_laws[i][j]]
            if b == 0:
                g.append(item)
                for l,k in enumerate(item):
                    u += sympy.Symbol('__s%d' % conservation_laws[i][l])    
                h.append(u-sympy.Symbol('C%d'%i))
                if verbose: print '  cycle%d'%i, 'is conserved'
        
        for i in h:
            constant_to_solve = [atom for atom in i.atoms(sympy.Symbol) if re.match(r'[C]',str(atom))]
            solution = sympy.solve(i, constant_to_solve ,implicit = True)
            solution_ready = solution[0]
            for q in solution_ready.atoms(sympy.Symbol): solution_ready = solution_ready.subs(q, y[0][str(q)])
            value_constants[constant_to_solve[0]] = solution_ready
            
        (self.conservation, self.conserve_var, self.value_conservation) = h, g, value_constants     
        return h, g, value_constants

    def passenger_equations(self):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(self.model)
        passenger_eqs = {}
        for i, j in enumerate(self.passengers):
            passenger_eqs[j] = self.model.odes[self.passengers[i]]
        return passenger_eqs

    def find_nearest_zero(self, array):
        idx = numpy.nanargmin(numpy.abs(array))
        return array[idx]

    # Make sure this is the "ignore:" y
    def pruned_equations(self, y, rho=1, ptge_similar=0.1, plot_prune = False):
        pruned_eqs = self.passenger_equations()
        equations  = copy.deepcopy(pruned_eqs)

        for j in equations:
            eq_monomials = equations[j].as_coefficients_dict().keys()         # Get monomials
            eq_monomials_iter = iter(eq_monomials)
            for l, m in enumerate(eq_monomials_iter):                         # Compares the monomials to find the pruned system
                m_ready = m                                                   # Monomial to compute with
                m_elim  = m                                                   # Monomial to save
                for p in self.param_values: m_ready = m_ready.subs(p, self.param_values[p]) # Substitute parameters
                second_mons_iter = iter(range(len(eq_monomials)))
                for k in second_mons_iter:
                    if (k+l+1) <= (len(eq_monomials)-1):
                        ble_ready = eq_monomials[k+l+1] # Monomial to compute with
                        ble_elim  = eq_monomials[k+l+1] # Monomial to save
                        for p in self.param_values: ble_ready = ble_ready.subs(p, self.param_values[p]) # Substitute parameters
                        args2 = []
                        args1 = []
                        variables_ble_ready = [atom for atom in ble_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                        variables_m_ready = [atom for atom in m_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                        f_ble = sympy.lambdify(variables_ble_ready, ble_ready, 'numpy' )
                        f_m = sympy.lambdify(variables_m_ready, m_ready, 'numpy' )
                        for uu,ll in enumerate(variables_ble_ready):
                            args2.append(y[:][str(ll)])
                        for w,s in enumerate(variables_m_ready):
                            args1.append(y[:][str(s)])
                        hey_pruned = numpy.log10(f_m(*args1)) - numpy.log10(f_ble(*args2))
                        
                        if plot_prune:
                            plt.figure()
                            plot1, = plt.plot(self.tspan[1:], f_m(*args1), 'r--', linewidth=5, label = str(m_elim).split('__')[1])
                            plot2, = plt.plot(self.tspan[1:], f_ble(*args2), label = str(ble_elim).split('__')[1])
                            plt.legend(handles = [plot1, plot2], loc=0)
                            plt.xlabel('time')
                            plt.ylabel('Particles/s')
                            plt.title(str(equations[j]))
                            
                        closest = self.find_nearest_zero(hey_pruned)
                        if closest > 0 and closest > rho:
                            pruned_eqs[j] = pruned_eqs[j].subs(ble_elim, 0)
                        elif closest < 0 and closest < -rho:
                            pruned_eqs[j] = pruned_eqs[j].subs(m_elim, 0) 
                            break
                        
                        else:pass
        plt.show()
#         for i in range(len(self.conservation)): #Add the conservation laws to the pruned system
#             pruned_eqs['cons%d'%i]=self.conservation[i]
        self.pruned = pruned_eqs
        
#         repeated_eq = 0
#         for idx in range (len(self.pruned.values())):
#             for n in range(idx+1,len(self.pruned)):
#                 comparing = sympy.simplify(self.pruned.values()[idx]-self.pruned.values()[n])
#                 print self.pruned.values()[idx],'minus',self.pruned.values()[n]
#                 if comparing ==0: repeated_eq +=1

        return pruned_eqs
     
    def solve_pruned(self):       
        solve_for = copy.deepcopy(self.passengers)
        eqs       = copy.deepcopy(self.pruned)
        eqs_l = eqs.values() 
        
        for idx, var_l in enumerate(self.conserve_var):
            if len(var_l) == 1:
                solve_for.append(var_l[0])
                eqs_l.append(self.conservation[idx])
        variables =  tuple(sympy.Symbol('__s%d' %var) for var in solve_for )
# Problem because there are more equations than variables
        for con in range(len(self.conservation)): #Add the conservation laws to the pruned system
            eqs_l.append(self.conservation[con])        
            sol = sympy.solve(eqs_l, variables, simplify = True)  
            if sol != []: break

        if isinstance(sol,dict):
            #TODO, ask Alex about this
            sol = [tuple(sol[v] for v in variables)]
#         sol=[]
        if len(sol) == 0:
            self.sol_pruned = { j:sympy.Symbol('__s%d'%j) for i, j in enumerate(solve_for) }
        else:
            self.sol_pruned = { j:sol[0][i] for i, j in enumerate(solve_for) }
 
#         solve_for = copy.deepcopy(self.passengers)
#         eqs       = copy.deepcopy(self.pruned)
#         eqs_l = eqs.values()   

#         expr_in_pruned = []
#         for p_eq in pruned_eqs.keys():
#             print p_eq, pruned_eqs[p_eq]
# #             eq_vars = [atom for atom in p_eq.atoms(sympy.Symbol) if not re.match(r'k', str(atom))]
# #             expr_in_pruned.append(list(set(eq_vars) & set(drivers_symbols)))
#         expr_flat = [item for sublist in expr_in_pruned for item in sublist] 


   
#         for idx, var_l in enumerate(self.conserve_var):
#             pasnger_in_cons_eq = []
#             pasnger_in_cons = list(set(var_l) & set(self.passengers))
#             solve_for = tuple(sympy.Symbol('__s%d'%var) for var in pasnger_in_cons)
#             pasnger_in_cons_eq = [eqs[pa] for pa in pasnger_in_cons]
#             pasnger_in_cons_eq.append(self.conservation[idx])
#             sol = sympy.solve(pasnger_in_cons_eq, solve_for, simplify=True, dict=True)
#             print self.passengers, var_l
#             print pasnger_in_cons_eq
#             print solve_for
#             print sol
#             if len(var_l) == 1:
#                 solve_for.append(var_l[0])
#         variables =  tuple(sympy.Symbol('__s%d' %var) for var in solve_for )
# # Problem because there are more equations than variables
#         sol = sympy.solve(eqs_l, variables, simplify=False, dict=False)
#         print eqs_l
#         print sol
#         if isinstance(sol,dict):
#             #TODO, ask Alex about this
#             sol = [tuple(sol[v] for v in variables)]
# #         sol=[]
# #         if len(sol) == 0:
# #             self.sol_pruned = { j:sympy.Symbol('__s%d'%j) for i, j in enumerate(solve_for) }
# #         else:
#         self.sol_pruned = { j:sol[0][i] for i, j in enumerate(solve_for) }
#         print self.sol_pruned          
        return self.sol_pruned

    def equations_to_tropicalize(self):
        idx = list(set(range(len(self.model.odes))) - set(self.sol_pruned.keys()))
        eqs = { i:self.model.odes[i] for i in idx }

        for l in eqs.keys(): #Substitutes the values of the algebraic system
#             for k in self.sol_pruned.keys(): eqs[l]=eqs[l].subs(sympy.Symbol('s%d' % k), self.sol_pruned[k])
            for q in self.value_conservation.keys(): eqs[l] = eqs[l].subs(q, self.value_conservation[q])
#         for i in eqs.keys():
#             for par in self.model.parameters: eqs[i] = sympy.simplify(eqs[i].subs(par.name, par.value))
        self.eqs_for_tropicalization = eqs

        return eqs
    
    def final_tropicalization(self):
        tropicalized = {}
        
        for j in sorted(self.eqs_for_tropicalization.keys()):
            if type(self.eqs_for_tropicalization[j]) == sympy.Mul: tropicalized[j] = self.eqs_for_tropicalization[j] #If Mul=True there is only one monomial
            elif self.eqs_for_tropicalization[j] == 0: print 'there are no monomials'
            else:            
                ar = sorted(self.eqs_for_tropicalization[j].as_coefficients_dict(), key=str) #List of the terms of each equation  
                asd=0 
                for l, k in enumerate(ar):
                    p = k
                    for f, h in enumerate(ar):
                       if k != h:
                          p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                    asd +=p
                tropicalized[j] = asd

        self.tropical_eqs = tropicalized
        return tropicalized

    def data_drivers(self, y):        
        tropical_system = self.final_tropicalization()
        trop_data = OrderedDict()
        signature_sp = {}
        driver_monomials = OrderedDict()

        for i in tropical_system.keys(): 
            signature = [0]*self.tspan[1:]
            mons_data = {}
            mons = sorted(tropical_system[i].as_coefficients_dict().items(),key=str)
            mons_matrix = numpy.zeros((len(mons),len(self.tspan[1:])), dtype=float)
            sign_monomial = tropical_system[i].as_coefficients_dict().values()
            spe_monomials = OrderedDict(sorted(self.model.odes[i].as_coefficients_dict().items(),key=str))
            driver_monomials[i] = spe_monomials
            
            for q, m_s in enumerate(mons):
#                 mon_inf = [None]*2
                j = list(m_s)
                jj = copy.deepcopy(j[0]) 
                for par in self.param_values: j[0]=j[0].subs(par, self.param_values[par])
                arg_f1 = []
                var_to_study = [atom for atom in j[0].atoms(sympy.Symbol) if not re.match(r'\d',str(atom))] #Variables of monomial 
                f1 = sympy.lambdify(var_to_study, j[0], modules = dict(Heaviside=_Heaviside_num, log=numpy.log10, Abs=numpy.abs)) 
                for va in var_to_study:
                   arg_f1.append(y[str(va)])    
#                 mon_inf[0]=f1(*arg_f1)
#                 mon_inf[1]=j[1]
                mons_data[str(jj).partition('*Heaviside')[0]] = f1(*arg_f1)
                mons_matrix[q] = f1(*arg_f1)
            for col in range(len(self.tspan[1:])):
                signature[col] = numpy.nonzero(mons_matrix[:,col])[0][0]
            signature_sp[i] = signature
            trop_data[i] = mons_data
        self.driver_signatures = signature_sp
        self.mon_names = driver_monomials
        self.tro_species = trop_data
        return trop_data 
    
    def visualization(self, driver_species=None):
        if driver_species is not None:
            species_ready = []
            for i in driver_species:
                if i in self.tro_species.keys(): species_ready.append(i)
                else: print 'specie' + ' ' + str(i) + ' ' + 'is not a driver'
        elif driver_species is None:
            raise Exception('list of driver species must be defined')
        
        if species_ready == []:
            raise Exception('None of the input species is a driver')
                     
        
        colors = itertools.cycle(["b", "g", "c", "m", "y", "k" ])
        
        sep = len(self.tspan)/2
      
        for sp in species_ready:
            si_flux = 0
            no_flux = 0
            monomials_dic = self.tro_species[sp].values()
            f = plt.figure()
            plt.subplot(211)
            monomials = []
            monomials_inf = self.mon_names[sp]
            for name in self.tro_species[sp].keys():
                m_value = self.tro_species[sp][name]
                x_concentration = numpy.nonzero(m_value)[0]
                monomials.append(name)            
                si_flux+=1
                x_points = [self.tspan[x] for x in x_concentration] 
                prueba_y = numpy.repeat(2*si_flux, len(x_concentration))
                if monomials_inf[sympy.sympify(name)] > 0 : plt.scatter(x_points[::int(math.ceil(len(self.tspan)/sep))], prueba_y[::int(math.ceil(len(self.tspan)/sep))],\
                                            color = next(colors), marker=r'$\uparrow$', s=numpy.array([m_value[k] for k in x_concentration])[::int(math.ceil(len(self.tspan)/sep))])
                if monomials_inf[sympy.sympify(name)] < 0 : plt.scatter(x_points[::int(math.ceil(len(self.tspan)/sep))], prueba_y[::int(math.ceil(len(self.tspan)/sep))], \
                                            color = next(colors), marker=r'$\downarrow$', s=numpy.array([m_value[k] for k in x_concentration])[::int(math.ceil(len(self.tspan)/sep))])
 
            y_pos = numpy.arange(2,2*si_flux+4,2)    
            plt.yticks(y_pos, monomials, size = 'x-small') 
            plt.ylabel('Monomials')
            plt.xlim(0, self.tspan[-1])
            plt.ylim(0,max(y_pos))
            plt.subplot(210)
            plt.plot(self.tspan[1:],self.y['__s%d'%sp][1:])
            plt.ylabel('Molecules')
            plt.xlabel('Time (s)')
            plt.suptitle('Tropicalization' + ' ' + str(self.model.species[sp]))
            plt.savefig('/home/carlos/Desktop/'+'s%d'%sp, format='jpg', bbox_inches='tight', dpi=400)
            plt.show()


#         plt.ylim(0, len(monomials)+1) 
        return f  

    def get_trop_data(self):
        return self.tro_species.keys()
    def get_passenger(self):
        return self.passengers
    def get_pruned_equations(self):
        return self.pruned
    
def run_tropical(model, tspan, parameters = None, sp_visualize = None):
    tr = Tropical(model)
    tr.tropicalize(tspan, parameters)
    if sp_visualize is not None:
        tr.visualization(driver_species=sp_visualize)
    return tr.get_passenger()


 
