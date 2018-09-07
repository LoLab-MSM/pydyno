from pysb import *

Model()

# Declaring the monomers of the model
Monomer('Arrestin', ['b1', 'b2', 'b3'])
Monomer('MKK4', ['b', 'state'], {'state': ['U', 'P']})
Monomer('MKK7', ['b', 'state'], {'state': ['U', 'P']})
Monomer('JNK3', ['b', 'threo', 'tyro'], {'threo': ['U', 'P'], 'tyro': ['U', 'P']})

# All kf parameters are in units of inverse microM*s
# All kr parameters are in units of inverse s
# All kcat parameters are units of inverse s
# The forward reaction is association; the reverse is disassociation

# Because the Ordinary differential equations derived from the mass action kinetics law requires rate
# constants instead of K_D values, K_D values are going to be converted into rate parameters (k_r/k_f).
# We are going to assume that the reaction rate k_f is 'the average enzyme', whereas the k_r would be allowed to vary

###### IMPORTANT INFO ABOUT JNK3

# pMKK4 with Arrestin-3, K_D = 347 microM, figure 1.B
Parameter('kf_pMKK4_Arr', 2)
Parameter('kr_pMKK4_Arr', 240)

# pMKK7 with Arrestin-3, K_D = 13 microM, figure 1.D
Parameter('kf_pMKK7_Arr', 2)
Parameter('kr_pMKK7_Arr', 26)

# Arrestin3-MKK4 bind to uuJNK3, K_D = 1.4 microM, figure 1.E
Parameter('kf_MKK4_Arr_bind_uuJNK3', 2)
Parameter('kr_MKK4_Arr_bind_uuJNK3', 2.8)

# Arrestin3 bind to upJNK3, K_D = 4.2 microM, figure 1.F
Parameter('kf_upJNK3BindArr', 2)
Parameter('kr_upJNK3BindArr', 20)

Parameter('kf_upJNK3_bind_Arr_MKK4', 2)
Parameter('kr_upJNK3_bind_Arr_MKK4', 8.4)

Parameter('kf_upJNK3_bind_Arr_MKK7', 2)
Parameter('kr_upJNK3_bind_Arr_MKK7', 8.4)

# Arrestin3 bind to puJNK3, K_D = 10.5 microM, figure 1.G
Parameter('kf_puJNK3BindArr', 2)
Parameter('kr_puJNK3BindArr', 20)

Parameter('kf_puJNK3_bind_Arr_MKK4', 2)
Parameter('kr_puJNK3_bind_Arr_MKK4', 21)

Parameter('kf_puJNK3_bind_Arr_MKK7', 2)
Parameter('kr_puJNK3_bind_Arr_MKK7', 21)

# # ppJNK3 with Arrestin-3, K_D = 220 microM, figure 1.H
Parameter('kf_ppJNK3_Arr', 2)
Parameter('kr_ppJNK3_Arr', 32)

# uuJNK3 binds Arrestin, K_D = 1.4 microM, figure 1.E
Parameter('kf_uuJNK3_Arr', 2)
Parameter('kr_uuJNK3_Arr', 2.2)

##### These are the parameters that are going to be calibrated

# Interacting JNK-docking Sites in MKK7 Promote Binding and
# Activation of JNK Mitogen-activated Protein Kinases* kd = 40 microm
Parameter('kf_MKK4BindArr_uuJNK3', 2)
Parameter('kr_MKK4BindArr_uuJNK3', 80)

# Parameter('kf_MKK4BindArr_puJNK3', 2)
# Parameter('kr_MKK4BindArr_puJNK3', 80)

# kd = 30
Parameter('kf_MKK7BindArr_JNK3', 2)
Parameter('kr_MKK7BindArr_JNK3', 60)

# Arrestin3-MKK7 bind to uuJNK3, K_D = 1.4 microM, Figure 1.E
Parameter('kf_MKK7_Arr_bind_uuJNK3', 2)
Parameter('kr_MKK7_Arr_bind_uuJNK3', 2.8)

# uJNK3 with MKK4
Parameter('kf_MKK4_uuJNK3', 2)
Parameter('kr_MKK4_uuJNK3', 80)

Parameter('kf_MKK4_puJNK3', 2)
Parameter('kr_MKK4_puJNK3', 80)

# uJNK3 with MKK7
# This is when MKK7 is bound
Parameter('kf_MKK7_uuJNK3', 2)
Parameter('kr_MKK7_uuJNK3', 60)

Parameter('kf_MKK7_upJNK3', 2)
Parameter('kr_MKK7_upJNK3', 60)

# MKK4, Arrestin JNK3 activation
Parameter('kcat_pMKK4_ArrJNK3', 1)

# MKK7, Arrestin JNK3 activation
Parameter('kcat_pMKK7_ArrJNK3', 1)

Parameter('keq_pMKK4_to_pMKK7', 1)
Parameter('keq_pMKK7_to_pMKK4', 1)

Parameter('kf_pJNK3_MKK4complex', 2)
Parameter('kr_pJNK3_MKK4complex', 80)

Parameter('kf_pJNK3_MKK7complex', 2)
Parameter('kr_pJNK3_MKK7complex', 60)

# Initial conditions
Parameter('Arrestin_0', 5)
Parameter('pMKK4_0', 0.05)
Parameter('pMKK7_0', 0.05)
Parameter('uuJNK3_0', 0.5959)
Parameter('puJNK3_0', 0)
Parameter('upJNK3_0', 0.00402)

Initial(Arrestin(b1=None, b2=None, b3=None), Arrestin_0)
Initial(MKK4(b=None, state='P'), pMKK4_0)
Initial(MKK7(b=None, state='P'), pMKK7_0)
Initial(JNK3(b=None, threo='U', tyro='U'), uuJNK3_0)
Initial(JNK3(b=None, threo='P', tyro='U'), puJNK3_0)
Initial(JNK3(b=None, threo='U', tyro='P'), upJNK3_0)

# Rules

# Arrestin interactions with MKK4/7 and JNK3

Rule('pMKK4BindArr', Arrestin(b1=None, b2=None, b3=None) + MKK4(b=None, state='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK4(b=2, state='P'), kf_pMKK4_Arr, kr_pMKK4_Arr)

Rule('pMKK7BindArr', Arrestin(b1=None, b2=None, b3=None) + MKK7(b=None, state='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK7(b=2, state='P'), kf_pMKK7_Arr, kr_pMKK7_Arr)

Rule('uuJNK3BindArr', Arrestin(b1=None, b2=None, b3=None) + JNK3(b=None, threo='U', tyro='U') |
     Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, threo='U', tyro='U'), kf_uuJNK3_Arr, kr_uuJNK3_Arr)

Rule('upJNK3BindArr', Arrestin(b1=None, b2=None, b3=None) + JNK3(b=None, threo='U', tyro='P') |
     Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, threo='U', tyro='P'), kf_upJNK3BindArr, kr_upJNK3BindArr)

Rule('puJNK3BindArr', Arrestin(b1=None, b2=None, b3=None) + JNK3(b=None, threo='P', tyro='U') |
     Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, threo='P', tyro='U'), kf_puJNK3BindArr, kr_puJNK3BindArr)

# MKK4 interactions with JNK3 in the arrestin scaffold

Rule('MKK4_ArrBinduuJNK3', Arrestin(b1=None, b2=2, b3=None) % MKK4(b=2, state='P') + JNK3(b=None, threo='U', tyro='U') |
     Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, threo='U', tyro='U'),
     kf_MKK4_Arr_bind_uuJNK3, kr_MKK4_Arr_bind_uuJNK3)

Rule('MKK4catJNK3Arr', Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, tyro='U') >>
     Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, tyro='P'),kcat_pMKK4_ArrJNK3)

# JNK3 has to get dissociated because otherwise it wouldnt be possible to have more pJNK3 than the value of MKK4 or MKK7
Rule('upJNK3Arr_MKK4_diss', Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3,threo='U', tyro='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK4(b=2, state='P') + JNK3(b=None, threo='U', tyro='P')
     , kr_upJNK3_bind_Arr_MKK4, kf_upJNK3_bind_Arr_MKK4)

Rule('puJNK3Arr_MKK4_diss', Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, threo='P', tyro='U') |
     Arrestin(b1=None, b2=2, b3=None) % MKK4(b=2, state='P') + JNK3(b=None, threo='P', tyro='U')
     , kr_puJNK3_bind_Arr_MKK4, kf_puJNK3_bind_Arr_MKK4)

Rule('ppJNK3Arr_MKK4_diss', Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, threo='P', tyro='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK4(b=2, state='P') + JNK3(b=None, threo='P', tyro='P')
     , kr_ppJNK3_Arr, kf_ppJNK3_Arr)

# MKK7 interactions with JNK3 in the arrestin scaffold

Rule('MKK7_ArrBindUUJNK3', Arrestin(b1=None, b2=2, b3=None) % MKK7(b=2, state='P') + JNK3(b=None, threo='U', tyro='U') |
     Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='U', tyro='U')
     , kf_MKK7_Arr_bind_uuJNK3, kr_MKK7_Arr_bind_uuJNK3)

Rule('MKK7catJNK3Arr', Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='U') >>
     Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='P'), kcat_pMKK7_ArrJNK3)

# JNK3 has to get dissociated because otherwise it wouldnt be possible to have more pJNK3 than the value of MKK4 or MKK7
Rule('puJNK3Arr_MKK7_diss', Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='P', tyro='U') |
     Arrestin(b1=None, b2=2, b3=None) % MKK7(b=2, state='P') + JNK3(b=None, threo='P', tyro='U')
     , kr_puJNK3_bind_Arr_MKK7, kf_puJNK3_bind_Arr_MKK7)

# Here we assume that the kinetics of puJNK3 disocciation for MKK4 are the same as for MKK7
Rule('upJNK3Arr_MKK7_diss', Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='U', tyro='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK7(b=2, state='P') + JNK3(b=None, threo='U', tyro='P')
     , kr_upJNK3_bind_Arr_MKK7, kf_upJNK3_bind_Arr_MKK7)

Rule('ppJNK3Arr_MKK7_diss', Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='P', tyro='P') |
     Arrestin(b1=None, b2=2, b3=None) % MKK7(b=2, state='P') + JNK3(b=None, threo='P', tyro='P')
     , kr_ppJNK3_Arr, kf_ppJNK3_Arr)

# MKK4/7 release from Arrestin complex
# We assume that MKK4/7 only binds to the Arr3:JNK3 complex when tyro/threo is unphosphorylated
Rule('MKK4DissArr_uuJNK3', Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, tyro='U') + MKK4(b=None, state='P')|
      Arrestin(b1=None, b2=2, b3=3) % JNK3(b=3, tyro='U') % MKK4(b=2, state='P'), kf_MKK4BindArr_uuJNK3, kr_MKK4BindArr_uuJNK3)

# Rule('MKK4DissArr_puJNK3', Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, threo='P', tyro='U') + MKK4(b=None, state='P')|
#       Arrestin(b1=None, b2=2, b3=3) % JNK3(b=3, threo='P', tyro='U') % MKK4(b=2, state='P'), kf_MKK4BindArr_puJNK3, kr_MKK4BindArr_puJNK3)

Rule('MKK7DissArr_JNK3', Arrestin(b1=None, b2=None, b3=3) % JNK3(b=3, threo='U') + MKK7(b=None, state='P')|
      Arrestin(b1=None, b2=2, b3=3) % JNK3(b=3, threo='U') % MKK7(b=2, state='P'), kf_MKK7BindArr_JNK3, kr_MKK7BindArr_JNK3)

# EquilibratePMKK4and7
Rule('EqpMKK4And7', Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, threo='U', tyro='P') + MKK7(b=None, state='P') >>
     Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='U', tyro='P') + MKK4(b=None, state='P'),
     keq_pMKK4_to_pMKK7)

Rule('EqpMKK7And4', Arrestin(b1=None, b2=2, b3=3) % MKK7(b=2, state='P') % JNK3(b=3, threo='P', tyro='U') + MKK4(b=None, state='P') >>
     Arrestin(b1=None, b2=2, b3=3) % MKK4(b=2, state='P') % JNK3(b=3, threo='P', tyro='U') + MKK7(b=None, state='P'),
     keq_pMKK7_to_pMKK4)

#### Direct interactions between MKK4/7 and JNK3

Rule('MKK4BinduuJNK3', MKK4(b=None, state='P') + JNK3(b=None, threo='U', tyro='U') |
     MKK4(b=1, state='P') % JNK3(b=1, threo='U', tyro='U')
     , kf_MKK4_uuJNK3, kr_MKK4_uuJNK3)

Rule('MKK4BindpuJNK3', MKK4(b=None, state='P') + JNK3(b=None, threo='P', tyro='U') |
     MKK4(b=1, state='P') % JNK3(b=1, threo='P', tyro='U')
     , kf_MKK4_puJNK3, kr_MKK4_puJNK3)

# phosphorylation should be the same with of witouh arrestin as arrestin
# only organizes spatially the molecules but doesnt enhance the catalysis
Rule('MKK4catJNK3', MKK4(b=1, state='P') % JNK3(b=1, tyro='U') >>
     MKK4(b=1, state='P') % JNK3(b=1, tyro='P'),kcat_pMKK4_ArrJNK3)

# JNK3 has to get dissociated because otherwise it wouldnt be possible to have more pJNK3 than the value of MKK4 or MKK7
Rule('pJNK3_MKK4complex_diss', MKK4(b=1, state='P') % JNK3(b=1, tyro='P') |
     MKK4(b=None, state='P') + JNK3(b=None, tyro='P'), kr_pJNK3_MKK4complex, kf_pJNK3_MKK4complex)

Rule('MKK7BinduuJNK3', MKK7(b=None, state='P') + JNK3(b=None, threo='U', tyro='U') |
     MKK7(b=1, state='P') % JNK3(b=1, threo='U', tyro='U'), kf_MKK7_uuJNK3, kr_MKK7_uuJNK3)

Rule('MKK7BindupJNK3', MKK7(b=None, state='P') + JNK3(b=None, threo='U', tyro='P') |
     MKK7(b=1, state='P') % JNK3(b=1, threo='U', tyro='P'), kf_MKK7_upJNK3, kr_MKK7_upJNK3)

Rule('MKK7catJNK3', MKK7(b=1, state='P') % JNK3(b=1, threo='U') >>
     MKK7(b=1, state='P') % JNK3(b=1, threo='P'), kcat_pMKK7_ArrJNK3)

# JNK3 has to get dissociated because otherwise it wouldnt be possible to have more pJNK3 than the value of MKK4 or MKK7
Rule('pJNK3_MKK7complex_diss', MKK7(b=1, state='P') % JNK3(b=1, threo='P') |
     MKK7(b=None, state='P') + JNK3(b=None, threo='P'), kr_pJNK3_MKK7complex, kf_pJNK3_MKK7complex)

# Observables
Observable('pTyr_jnk3', JNK3(tyro='P'))
Observable('pThr_jnk3', JNK3(threo='P'))
Observable('all_jnk3', JNK3(tyro='P', threo='P'))
