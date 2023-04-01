from pack.data.CSV import read_csv


def i_val(data, csv: bool = True):
    if csv:
        val = {}
        for j in data:
            val[j] = data[j][0]
    else:
        val = data

    x0 = list(val.values())

    return val, x0


def variables(dic):
    return list(dic.keys())


dt=(i_val(read_csv('D:/Mestrado/2ano/Tese/code/datasets/initial_val.csv')))
print(variables(dt[0]))
print(dt[1])
# x0 = [Q, Qw, Qr, Qef, Qr_p, V_a, A_s, h3, h4, r_p, X_I, X_Ir, X_Ief, S_Sent, S_S,
#       S_Oent, S_NOent, S_NO, X_BHent, X_BH, X_BHr, X_BHef, X_Sent,
#       X_S, X_Sr, X_Sef, X_BAent, X_BA, X_BAr, X_BAef, S_NHent, S_NH, X_Pent, X_P, X_Pr,
#       X_Pef, S_NDent, S_ND, X_NDent, X_ND, X_NDr, X_NDef,
#       G_s, S_alkent, S_alk, SSI, SSIef, SSIr, ST_1, ST_2, ST_3, ST_4, ST_5, ST_6, ST_7, ST_8, ST_9, ST_10,
#       v_dn, v_up, v_s_1, v_s_2, v_s_3, v_s_4, v_s_5, v_s_6, v_s_7, v_s_8, v_s_9, v_s_10, J_1, J_2, J_3, J_4, J_5,
#       J_6, J_7, J_8, J_9, J_10, HRT, KLa, r, Sent,
#       S, Xent, X, Xr, Xef, CODent, COD, CODr,
#       CODef, VSSent, VSS, VSSr, VSSef, TSSent, TSS, TSSr, TSSef, BODent, BOD, BODr, BODef,
#       TKNent, TKN, TKNr, TKNef, Nent, N, Nr, Nef, h, S_O]
