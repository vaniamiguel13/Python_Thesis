from pack.init_val import i_val
from pack.data.CSV import read_csv
from pack.parameters.parameters import parameters


def bounds(init, param):

    globals().update(init)
    globals().update(param)

    LBUB = [
        [Qinf, 5 * Qinf],  # Q
        [1, Qinf],  # Qw
        [.5 * Qinf, 5 * Qinf],  # Qr
        [.5 * Qinf, 2 * Qinf],  # Qef
        [1, 2 * Q_P],  # Qr_p
        [100, 10000],  # V_a
        [10, 1000],  # A_s
        [0.2, 5],  # h3
        [0.2, 5],  # h4
        [0.5, 2],  # r_p
        [10, 10000],  # X_I
        [10, 10000],  # X_Ir
        [0.01, 100],  # X_Ief
        [10, 1000],  # S_Sent
        [10, 100],  # S_S
        [0.001, 10],  # S_Oent
        [0, 100],  # S_NOent
        [0, 100],  # S_NO
        [0, 10000],  # X_BHent
        [1, 10000],  # X_BH
        [1, 10000],  # X_BHr
        [0.01, 1000],  # X_BHef
        [1, 10000],  # X_Sent
        [1, 10000],  # X_S
        [1, 10000],  # X_Sr
        [0.01, 500],  # X_Sef
        [0, 500],  # X_BAent
        [0, 500],  # X_BA
        [0.01, 10000],  # X_BAr
        [1e-6, 1000],  # X_BAef
        [0.1, 100],  # S_NHent
        [0.001, 100],  # S_NH
        [5, 10000],  # X_Pent
        [5, 10000],  # X_P
        [5, 10000],  # X_Pr
        [0.001, 100],  # X_Pef
        [0.01, 10],  # S_NDent
        [0.01, 50],  # S_ND
        [0.01, 1000],  # X_NDent
        [0.01, 1000],  # X_ND
        [0.01, 1000],  # X_NDr
        [0.001, 50],  # X_NDef
        [100, 200000],  # G_s
        [6, 8],  # S_alkent
        [6, 8],  # S_alk
        [10, 10000],  # SSI
        [0.1, 50],  # SSIef
        [10, 10000],  # SSIr
        [0.1, 10000],  # ST1
        [0.1, 10000],  # ST2
        [0.1, 10000],  # ST3
        [0.1, 10000],  # ST4
        [0.1, 10000],  # ST5
        [0.1, 10000],  # ST6
        [0.1, 10000],  # ST7
        [0.1, 10000],  # ST8
        [0.1, 10000],  # ST9
        [0.1, 10000],  # ST10
        [0.01, 10000],  # v_dn
        [0.01, 10000],  # v_up
        [0.01, 10000],  # v_s1
        [0.01, 10000],  # v_s2
        [0.01, 10000],  # v_s3
        [0.01, 10000],  # v_s4
        [0.01, 10000],  # v_s5
        [0.01, 10000],  # v_s6
        [0.01, 10000],  # v_s7
        [0.01, 10000],  # v_s8
        [0.01, 10000],  # v_s9
        [0.01, 10000],  # v_s10
        [0.01, 10000],  # J1
        [0.01, 10000],  # J2
        [0.01, 10000],  # J3
        [0.01, 10000],  # J4
        [0.01, 10000],  # J5
        [0.01, 10000],  # J6
        [0.01, 100000],  # J7
        [0.01, 100000],  # J8
        [0.01, 100000],  # J9
        [0.01, 1000000],  # J10
        [0.05, 2],  # HRT
        [0.01, 300],  # Kla
        [0.5, 2],  # r
        [0.1, 1000],  # Sent
        [0.1, 1000],  # S
        [0.1, 10000],  # Xent
        [0.1, 10000],  # X
        [0.1, 10000],  # Xr
        [0.1, 1000],  # Xef
        [0.1, 10000],  # CODent
        [0.1, 10000],  # COD
        [0.1, 10000],  # CODr
        [0.1, COD_law],  # CODef
        [0.1, 10000],  # VSSent
        [0.1, 10000],  # VSS
        [0.1, 10000],  # VSSr
        [0.1, 100],  # VSSef
        [0.1, 10000],  # TSSent
        [0.1, 10000],  # TSS
        [0.1, 10000],  # TSSr
        [0.1, TSS_law],  # TSSef
        [0.1, 10000],  # BODent
        [0.1, 10000],  # BOD
        [0.1, 10000],  # BODr
        [0.1, 100],  # BODef
        [0.1, 1000],  # TKNent
        [0.1, 1000],  # TKN
        [0.1, 1000],  # TKNr
        [0.1, 100],  # TKNef
        [0.1, 1000],  # Nent
        [0.1, 1000],  # N
        [0.1, 1000],  # Nr
        [0.1, N_law],  # Nef
        [0.1, 5],  # h
        [2, 10]]  # S_O

    LB = [row[0] for row in LBUB]
    UB = [row[1] for row in LBUB]

    return LB, UB
#
#
# print(bounds(i_val(read_csv('D:/Mestrado/2ano/Tese/code/datasets/initial_val.csv'))[0],
#                    parameters(read_csv('D:/Mestrado/2ano/Tese/code/param.csv'))))[0]
