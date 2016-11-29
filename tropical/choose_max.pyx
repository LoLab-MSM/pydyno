import math
import pandas as pd

def choose_max3(pd_series, diff_par, mon_comb, type_sign):
    """

    :param type_sign: Type of signature. It can be 'consumption' or 'consumption'
    :param mon_comb: combinations of monomials that produce certain species
    :param pd_series: Pandas series whose axis labels are the monomials and the data is their values at a specific
    time point
    :param diff_par: Parameter to define when a monomial is larger
    :return: monomial or combination of monomials that dominate at certain time point
    """
    if type_sign == 'production':
        monomials = pd_series[pd_series > 0]
        value_to_add = 1e-100
        sign = 1
        ascending = False
    elif type_sign == 'consumption':
        monomials = pd_series[pd_series < 0]
        value_to_add = -1e-100
        sign = -1
        ascending = True
    else:
        raise Exception('Wrong type_sign')

    # chooses the larger monomial or combination of monomials that satisfy diff_par
    largest_prod = 'ND'
    for comb in mon_comb.keys():
        # comb is an integer that represents the number of monomials in a combination
        if comb == mon_comb.keys()[-1]:
            break

        if len(mon_comb[comb].keys()) == 1:
            largest_prod = mon_comb[comb].keys()[0]
            break

        monomials_values = {}
        for idx in mon_comb[comb].keys():
            value = 0
            for j in mon_comb[comb][idx]:
                # j(reversible) might not be in the prod df because it has a negative value
                if j not in list(monomials.index):
                    value += value_to_add
                else:
                    value += monomials.loc[j]
            monomials_values[idx] = value

        foo2 = pd.Series(monomials_values).sort_values(ascending=ascending)
        comb_largest = mon_comb[comb][list(foo2.index)[0]]
        for cm in list(foo2.index):
            # Compares the largest combination of monomials to other combinations whose monomials that are not
            # present in comb_largest
            if len(set(comb_largest) - set(mon_comb[comb][cm])) == len(comb_largest):
                value_prod_largest = math.log10(sign * foo2.loc[list(foo2.index)[0]])
                if abs(value_prod_largest - math.log10(sign * foo2.loc[cm])) > diff_par and value_prod_largest > -5:
                    largest_prod = list(foo2.index)[0]
                    break
        if largest_prod != 'ND':
            break
    return largest_prod