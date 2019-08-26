from pysb import Model, Parameter, Initial, Monomer, Rule, Observable, MatchOnce
import pysb.macros as macros

Model()

Monomer('L', ['bf'])
Monomer('R', ['bf'])
Monomer('DISC', ['bf'])
Monomer('C8', ['bf', 'state'], {'state': ['pro', 'A']})
Monomer('Bid', ['bf', 'state'], {'state': ['U', 'T', 'M']})
Monomer('Bax', ['bf', 's1', 's2', 'state'], {'state': ['C', 'M', 'A']})
Monomer('Bcl2', ['bf'])
Monomer('Puma', ['bf', 'state'], {'state': ['C', 'M']})
Monomer('A1', ['bf'])
Monomer('Noxa', ['bf'])
Monomer('Smac', ['bf', 'state'], {'state': ['M', 'C']})

# Monomer('C8')
# Monomer('C3')
# Monomer('Smac')
# Monomer('Parp')

Parameter('k3', 0.000015)
Parameter('k16', 0.00001)
Parameter('k2', 0.0004)
Parameter('kf2', 0.002)
Parameter('k4', 0.0002)
Parameter('kf4', 0.003)
Parameter('k12', 0.0002)
Parameter('kf12', 0.003)
Parameter('k14', 0.0002)
Parameter('kf14', 0.003)
Parameter('k18', 0.0005)
Parameter('kf18', 0.002)
Parameter('kBidon', 0.0002)
Parameter('kBidoff', 0.00002)
Parameter('kBaxon', 0.0002)
Parameter('kBaxoff', 0.00002)
Parameter('kPumaon', 0.0002)
Parameter('kPumaoff', 0.00002)
Parameter('k24', 0.0004)
Parameter('kf24', 0.004)
Parameter('k25', 0.0005)
Parameter('kf25', 0.002)
Parameter('k26', 0.0002)
Parameter('kf26', 0.001)
Parameter('k27', 0.0002)
Parameter('kf27', 0.003)
Parameter('bind_L_R_to_LR_kf', 4e-07)
Parameter('bind_L_R_to_LR_kr', 0.001)
Parameter('convert_LR_to_DISC_kc', 1e-05)
Parameter('bind_DISC_C8pro_to_DISCC8pro_kf', 1e-06)
Parameter('bind_DISC_C8pro_to_DISCC8pro_kr', 0.001)
Parameter('catalyze_DISCC8pro_to_DISC_C8A_kc', 1.0)
Parameter('bind_C8A_BidU_to_C8ABidU_kf', 1e-06)
Parameter('bind_C8A_BidU_to_C8ABidU_kr', 0.001)
Parameter('catalyze_C8ABidU_to_C8A_BidT_kc', 1.0)
Parameter('pore_transport_complex_BaxA_4_SmacM_kf', 2.857143e-05)
Parameter('pore_transport_complex_BaxA_4_SmacM_kr', 0.001)
Parameter('pore_transport_dissociate_BaxA_4_SmacC_kc', 10.0)

# Initial conditions
Parameter('L_0', 3000.0)
Parameter('R_0', 200.0)
Parameter('C8_0', 20000.0)
Parameter('Bid_0', 40000.0)
Parameter('Bcl2_0', 20000.0)
Parameter('Noxa_0', 1000.0)
Parameter('Smac_0', 100000.0)
Parameter('Bax_0', 80000.0)
Parameter('A1_0', 1000)

Initial(L(bf=None), L_0)
Initial(R(bf=None), R_0)
Initial(C8(bf=None, state='pro'), C8_0)
Initial(Bid(bf=None, state='U'), Bid_0)
Initial(Bax(bf=None, s1=None, s2=None, state='C'), Bax_0)
Initial(Bcl2(bf=None), Bcl2_0)
Initial(Noxa(bf=None), Noxa_0)
Initial(Smac(bf=None, state='M'), Smac_0)
Initial(A1(bf=None), A1_0)

Rule('bind_L_R_to_LR', L(bf=None) + R(bf=None) | L(bf=1) % R(bf=1), bind_L_R_to_LR_kf, bind_L_R_to_LR_kr)
Rule('convert_LR_to_DISC', L(bf=1) % R(bf=1) >> DISC(bf=None), convert_LR_to_DISC_kc)
Rule('bind_DISC_C8pro_to_DISCC8pro', DISC(bf=None) + C8(bf=None, state='pro') | DISC(bf=1) % C8(bf=1, state='pro'), bind_DISC_C8pro_to_DISCC8pro_kf, bind_DISC_C8pro_to_DISCC8pro_kr)
Rule('catalyze_DISCC8pro_to_DISC_C8A', DISC(bf=1) % C8(bf=1, state='pro') >> DISC(bf=None) + C8(bf=None, state='A'), catalyze_DISCC8pro_to_DISC_C8A_kc)
Rule('bind_C8A_BidU_to_C8ABidU', C8(bf=None, state='A') + Bid(bf=None, state='U') | C8(bf=1, state='A') % Bid(bf=1, state='U'), bind_C8A_BidU_to_C8ABidU_kf, bind_C8A_BidU_to_C8ABidU_kr)
Rule('catalyze_C8ABidU_to_C8A_BidT', C8(bf=1, state='A') % Bid(bf=1, state='U') >> C8(bf=None, state='A') + Bid(bf=None, state='T'), catalyze_C8ABidU_to_C8A_BidT_kc)


macros.equilibrate(Bid(bf=None, state='T'), Bid(bf=None, state='M'), [kBidon, kBidoff])  # This doesn't take into account the change in volume from the change in compartments
Rule('tBid_bind_A1', Bid(bf=None, state='T') + A1(bf=None) | Bid(bf=1, state='T') % A1(bf=1), k26, kf26)

Rule('mBid_bind_Bcl2', Bid(bf=None, state='M') + Bcl2(bf=None) | Bid(bf=1, state='M') % Bcl2(bf=1), k2, kf2)

macros.catalyze_one_step(Bid(bf=None, state='M'), Bax(bf=None, s1=None, s2=None, state='C'),
                         Bax(bf=None, s1=None, s2=None, state='A'), k3)
macros.equilibrate(Bax(bf=None, s1=None, s2=None, state='A'), Bax(bf=None, s1=None, s2=None, state='M'), [kBaxon, kBaxoff])
Rule('aBax_bind_A1', Bax(bf=None, s1=None, s2=None, state='A') + A1(bf=None) |
     Bax(bf=1, s1=None, s2=None, state='A') % A1(bf=1), k24, kf24)
macros.catalyze_one_step(Bax(bf=None, s1=None, s2=1, state='M') % Bax(bf=None, s1=1, s2=None, state='M'),
                         Bax(bf=None, s1=None, s2=None, state='C'), Bax(bf=None, s1=None, s2=None, state='A'), k16)

Rule('mBax_bind_Bcl2', Bax(bf=None, s1=None, s2=None, state='M') + Bcl2(bf=None) |
     Bax(bf=1, s1=None, s2=None, state='M') % Bcl2(bf=1), k4, kf4)
Rule('Bcl2_bind_puma', Bcl2(bf=None) + Puma(bf=None, state='M') | Bcl2(bf=1) % Puma(bf=1, state='M'), k18, kf18)

Rule('mBax_bind_mBax', Bax(bf=None, s1=None, s2=None, state='M') + Bax(bf=None, s1=None, s2=None, state='M') |
     Bax(bf=None, s1=None, s2=1, state='M') % Bax(bf=None, s1=1, s2=None, state='M'), k12, kf12)
Rule('mBax2_bind_mBax2', Bax(bf=None, s1=None, s2=1, state='M') % Bax(bf=None, s1=1, s2=None, state='M') +
     Bax(bf=None, s1=None, s2=3, state='M') % Bax(bf=None, s1=3, s2=None, state='M') |
     MatchOnce(Bax(bf=None, s1=2, s2=1, state='M') % Bax(bf=None, s1=1, s2=4, state='M') % Bax(bf=None, s1=4, s2=3, state='M') % Bax(bf=None, s1=3, s2=2, state='M')), k14, kf14)

macros.equilibrate(Puma(bf=None, state='C'), Puma(bf=None, state='M'), [kPumaon, kPumaoff])
Rule('Puma_bind_A1', Puma(bf=None, state='C') + A1(bf=None) | Puma(bf=1, state='C') % A1(bf=1), k25, kf25)

Rule('A1_bind_Noxa', A1(bf=None) + Noxa(bf=None) | A1(bf=1) % Noxa(bf=1), k27, kf27)

Rule('pore_transport_complex_BaxA_4_SmacM', MatchOnce(Bax(bf=None, s1=2, s2=1, state='M') % Bax(bf=None, s1=1, s2=4, state='M') % Bax(bf=None, s1=4, s2=3, state='M') % Bax(bf=None, s1=3, s2=2, state='M')) + Smac(bf=None, state='M') | MatchOnce(Bax(bf=5, s1=2, s2=1, state='M') % Bax(bf=None, s1=1, s2=4, state='M') % Bax(bf=None, s1=4, s2=3, state='M') % Bax(bf=None, s1=3, s2=2, state='M')) % Smac(bf=5, state='M'), pore_transport_complex_BaxA_4_SmacM_kf, pore_transport_complex_BaxA_4_SmacM_kr)
Rule('pore_transport_dissociate_BaxA_4_SmacC', MatchOnce(Bax(bf=5, s1=2, s2=1, state='M') % Bax(bf=None, s1=1, s2=4, state='M') % Bax(bf=None, s1=4, s2=3, state='M') % Bax(bf=None, s1=3, s2=2, state='M')) % Smac(bf=5, state='M') >> MatchOnce(Bax(bf=None, s1=2, s2=1, state='M') % Bax(bf=None, s1=1, s2=4, state='M') % Bax(bf=None, s1=4, s2=3, state='M') % Bax(bf=None, s1=3, s2=2, state='M')) + Smac(bf=None, state='C'), pore_transport_dissociate_BaxA_4_SmacC_kc)

Observable('mBid', Bid(state='M'))
Observable('cSmac', Smac(state='C'))