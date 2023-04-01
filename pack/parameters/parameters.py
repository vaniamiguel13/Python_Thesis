from pack.data.CSV import read_csv


def parameters(data, CSV: bool = True):
    if CSV:
        param = {}
        for j in data:
            param[j] = data[j][0]
    else:
        param = data

    param['dens'] = 999.96 * (2.29e-2 * param['T']) - (5.44e-3 * param['T'] ** 2)
    param['TSSr_max'] = 1200 * 1000 / param['IVLD']
    param['TSSr_max_p'] = (1200 / param['IVLD'] + 2) * 1000
    param['Henry'] = (708 * param['T']) + 25700
    param['SOST'] = 1777.8 * param['beta'] * param['dens'] * param['P_O2'] / param['Henry']
    param['TKNinf'] = param['S_NHinf'] + param['S_NDinf'] + param['X_NDinf'] + param['i_XB'] * (
            param['X_BHinf'] + param['X_BAinf']) + param['i_XP'] * (param['X_Pinf'] + param['X_Iinf'])
    param['Ninf'] = param['TKNinf'] + param['S_NOinf']
    param['Xinf'] = param['X_Iinf'] + param['X_Sinf'] + param['X_BHinf'] + param['X_BAinf'] + param['X_Pinf']
    param['Sinf'] = param['S_I'] + param['S_Sinf']
    param['CODinf'] = param['Xinf'] + param['Sinf']
    param['VSSinf'] = param['Xinf'] / param['icv']
    param['TSSinf'] = param['VSSinf'] + param['X_IIinf']
    param['BODinf'] = param['f_BOD'] * (param['S_Sinf'] + param['X_Sinf'] + param['X_BHinf'] + param['X_BAinf'])

    globals().update(param)

    return param


dt_param=(parameters(read_csv('D:/Mestrado/2ano/Tese/code/param.csv')))
par=list(dt_param.values())
print(par)
# print(parameters(dic, True))
