"""The model plays important role in the initial stage of cancer cell invasion.
Molecule MT1-MMP exists on the surface of cancer cell. This molecule works with
MMP2 and TIMP2 to degrade extracellular matrix (ECM) and then open the way for 
cancer cell to leave the primary state to metastasize to distant part of the body.
MT1-MMP model talks about binding reactions between the three components,
MT1-MMP, MMP, and TIMP2. There is one component including in the system, say abcc,
that is believe to be a component that will make the ECM degradation happened.
SO, in the simulation, we always want to know the value of abcc at the equilibrium state"""
#gfgbfdbdfb
from pysb import *

def monomer_abc_model():
    """Let a, b, and c be MMP2, TIMP2, and MT1-MMP, respectivelly.
    Monomer a has only one binding site. Each of monomer b and c has two sites"""
    Monomer('a',['a1'])
    Monomer('b',['b1','b2'])
    Monomer('c',['c1','c2'])

def rate_constant_abc_model():
    #default rate constants
    Parameter('kab', 2.1e7)
    Parameter('kbc', 2.74e6)
    Parameter('lbc', 2e-4)
    Parameter('kcc', 2*2e6)
    Parameter('lcc', 1e-2)

def rule_original_abc_model():
    """Monomer a can bind b. Monomer b can bind to monomer a and c on each sites.
    Monomer c can form dimer and bind b."""
    #binding criteria : (ab) b1 with a1, (bc) b2 with c1,(cc) c2 with itself
    Rule('ab', a(a1=None) + b(b1=None) >> a(a1=1)%b(b1=1), kab)
    Rule('bc', b(b2=None) + c(c1=None) <> b(b2=1)%c(c1=1), kbc, lbc)
    Rule('cc', c(c2=None) + c(c2=None) <> c(c2=1)%c(c2=1), kcc, lcc)
    
def rule_abremoved_abc_model():
    #model knockout 1
    #remove ab's supplies
    #remove forward reaction of ab + {c, cc, bcc, abcc}
    
    Rule('ab', a(a1=None) + b(b1=None) >> a(a1=1)%b(b1=1), kab)
    Rule('bc', b(b1=None, b2=None) + c(c1=None) <> b(b1=None, b2=1)%c(c1=1), kbc, lbc)
    Rule('cc', c(c2=None) + c(c2=None) <> c(c2=1)%c(c2=1), kcc, lcc)
    Rule('abc', a(a1=1)%b(b1=1, b2=2)%c(c1=2) >> a(a1=1)%b(b1=1, b2=None) + c(c1=None), lbc)
    Rule('abcc', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3) >> a(a1=1)%b(b1=1, b2=None) + c(c1=None, c2=3)%c(c2=3), lbc)
    Rule('abccb', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4) >> a(a1=1)%b(b1=1, b2=None) + c(c1=None,c2=3)%c(c2=3, c1=4)%b(b2=4), lbc)
    Rule('abccba', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=5)%a(a1=5) >> a(a1=1)%b(b1=1, b2=None) + c(c1=None, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=5)%a(a1=5), lbc)

def rule_abcremoved1_abc_model():
    #model knockout 2
    #remove abc's supplies
    #remove forward reaction of abc + {c, bc, abc} and backward reaction of ab + c <> abc
    Rule('ab', a(a1=None) + b(b1=None) >> a(a1=1)%b(b1=1), kab)
    Rule('bc', b(b2=None) + c(c1=None) <> b(b2=1)%c(c1=1), kbc, lbc)
    Rule('cc', c(c1=None, c2=None) + c(c2=None, c1=None) <> c(c1=None,c2=1)%c(c2=1,c1=None), kcc, lcc)
    Rule('bcc', b(b1=None, b2=1)%c(c1=1, c2=None) + c(c1=None, c2=None) <> b(b1=None, b2=1)%c(c1=1, c2=2)%c(c1=None, c2=2), kcc, lcc)
    Rule('bccb', b(b1=None, b2=1)%c(c1=1, c2=None) + c(c1=1, c2=None)%b(b1=None, b2=1) <> b(b1=None, b2=1)%c(c1=1, c2=2)%c(c1=3, c2=2)%b(b1=None, b2=3), kcc, lcc)
    Rule('abcc', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=None) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=None), lcc)
    Rule('abccb', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=None) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=4)%b(b2=4, b1=None), lcc)
    Rule('abccba', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=5)%a(a1=5) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=4)%b(b2=4, b1=5)%a(a1=5), lcc)

def rule_abcremoved2_abc_model():
    #model knockout 2
    #remove abc's supplies
    #remove forward reaction of abc + {c, bc, abc} and backward reaction of ab + c <> abc
    Rule('ab', a(a1=None) + b(b1=None) >> a(a1=1)%b(b1=1), kab)
    Rule('bc', b(b1=None, b2=None) + c(c1=None) <> b(b1=None, b2=1)%c(c1=1), kbc, lbc)
    Rule('cc', c(c1=None, c2=None) + c(c2=None, c1=None) <> c(c1=None,c2=1)%c(c2=1,c1=None), kcc, lcc)
    Rule('abc_ab', a(a1=1)%b(b1=1, b2=None) + c(c1=None) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2), kbc) # ab + c -> abc
    Rule('abcc_ab', a(a1=1)%b(b1=1, b2=None) + c(c1=None, c2=2)%c(c2=2, c1=None) <> a(a1=1)%b(b1=1, b2=3)%c(c1=3, c2=2)%c(c2=2, c1=None), kbc, lbc) # ab + cc <> abcc
    Rule('abccb_ab', a(a1=1)%b(b1=1, b2=None) + c(c1=None, c2=2)%c(c2=2, c1=3)%b(b1=None, b2=3) <> a(a1=1)%b(b1=1, b2=3)%c(c1=3, c2=2)%c(c2=2, c1=4)%b(b1=None, b2=4), kbc, lbc) # ab +ccb <> abccb
    Rule('abccba_ab', a(a1=1)%b(b1=1, b2=None) + c(c1=None, c2=2)%c(c2=2, c1=3)%b(b1=4, b2=3)%a(a1=4) <> a(a1=1)%b(b1=1, b2=3)%c(c1=3, c2=2)%c(c2=2, c1=4)%b(b1=5, b2=4)%a(a1=5), kbc, lbc) # ab + ccba <> abccba
    Rule('bcc', b(b1=None, b2=1)%c(c1=1, c2=None) + c(c1=None, c2=None) <> b(b1=None, b2=1)%c(c1=1, c2=2)%c(c1=None, c2=2), kcc, lcc)
    Rule('bccb', b(b1=None, b2=1)%c(c1=1, c2=None) + c(c1=1, c2=None)%b(b1=None, b2=1) <> b(b1=None, b2=1)%c(c1=1, c2=2)%c(c1=3, c2=2)%b(b1=None, b2=3), kcc, lcc)
    Rule('abcc', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=None) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=None), lcc)
    Rule('abccb', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=None) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=4)%b(b2=4, b1=None), lcc)
    Rule('abccba', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=5)%a(a1=5) >> a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None) + c(c2=None, c1=4)%b(b2=4, b1=5)%a(a1=5), lcc)

def initial_condition_abc_model():
    #from the data
    Initial(a(a1=None), Parameter('ao', 1e-6))
    Initial(b(b1=None, b2=None), Parameter('bo', 1.57e-7)) #1.57e-7
    Initial(c(c1=None, c2=None), Parameter('co', 1e-6))

def observe_abc_model():
    """From the rules we have, in total, 12 components"""
    Observable('U0', c(c1=None, c2=None) + b(b1=None, b2=1) % c(c1=1, c2=None) + a(a1=1) % b(b1=1, b2=2) % c(c1=2, c2=None))
    Observable('U1', b(b1=None, b2=None) + a(a1=1) % b(b1=1, b2=2) % b(b1=None, b2=3) % c(c1=2, c2=4) % c(c1=3, c2=4) + b(b1=None, b2=1) % c(c1=1, c2=None) + b(b1=None, b2=1) % c(c1=1, c2=2) % c(c1=None, c2=2) + b(b1=None, b2=1) % b(b1=None, b2=2) % c(c1=1, c2=3) % c(c1=2, c2=3))
    Observable('U4', b(b1=None, b2=None) + a(a1=1) % b(b1=1, b2=None))
    Observable('U2', c(c1=None, c2=None) + b(b1=None, b2=1) % c(c1=1, c2=2) % c(c1=None, c2=2) + a(a1=1) % b(b1=1, b2=2) % c(c1=2, c2=3) % c(c1=None, c2=3) + c(c1=None, c2=1) % c(c1=None, c2=1))
    Observable('U3', a(a1=None))
    Observable('ta', a(a1=None))
    Observable('tb', b(b1=None, b2=None))
    Observable('tc', c(c1=None, c2=None))
    Observable('tab', a(a1=1)%b(b1=1, b2=None))
    Observable('tbc', b(b1=None, b2=1)%c(c1=1, c2=None))
    Observable('tcc', c(c1=None, c2=1)%c(c1=None, c2=1))
    Observable('tabc', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=None))
    Observable('tbcc', b(b1=None, b2=1)%c(c1=1, c2=2)%c(c1=None, c2=2))
    Observable('tabcc', a(a1=1)%b(b1=1,b2=2)%c(c1=2,c2=3)%c(c1=None,c2=3))
    Observable('tbccb', b(b1=None, b2=1)%c(c1=1, c2=2)%c(c2=2, c1=3)%b(b2=3, b1=None))
    Observable('tabccb', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=None))
    Observable('tabccba', a(a1=1)%b(b1=1, b2=2)%c(c1=2, c2=3)%c(c2=3, c1=4)%b(b2=4, b1=5)%a(a1=5))

def return_model(model_type):
    Model()
    monomer_abc_model()
    rate_constant_abc_model()
    initial_condition_abc_model()
    if model_type=='original':
        rule_original_abc_model()
    if model_type=='abremoved':
        rule_abremoved_abc_model()
    if model_type=='abc1removed':
        rule_abcremoved1_abc_model()
    if model_type=='abc2removed':
        rule_abcremoved2_abc_model()
    observe_abc_model()
    return model
Model()
monomer_abc_model()
rate_constant_abc_model()
initial_condition_abc_model()
rule_original_abc_model()
  