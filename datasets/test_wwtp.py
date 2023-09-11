import numpy as np
from HGPSAL.HGPSAL.AL_HGPSAL import HGPSAL
from MEGA.MEGACON.MegaCon import MegaCon
from pack.parameters.parameters import parameters
from pack.data.CSV import read_csv

x0= [4000, 100, 2000, 1900, 100, 1000, 1000, 2, 0.5, 0.5, 727.3, 950.571, 1e-05, 50, 10, 1, 1e-06, 1e-06, 0.2, 350, 711.2, 1e-05, 3000, 350, 806.714, 1e-05, 1e-05, 1e-06, 1.9e-6, 1e-05, 10, 7.5, 1500, 90, 174.52, 1e-05, 0.5, 0.5, 200, 20, 20, 0.5, 10000, 7, 7, 1500, 0.1, 3500, 10, 20, 40, 70, 200, 300, 350, 350, 2000, 4000, 21, 19, 7.941, 11.77, 15.92, 22.09, 34.25, 67.15, 179.1, 179.1, 179.1, 116.6, 0.08737, 0.1836, 0.3291, 0.6257, 1.505, 6.057, 59.05, 59.05, 59.06, 0, 3.5, 100, 1, 80, 50, 5000, 1000, 4440, 1e-04, 5500, 1600, 5500, 80, 3500, 1050, 3500, 1e-06, 5000, 1800, 5000, 10, 2000, 500, 2000, 30, 350, 106, 350, 10, 350, 106, 350, 15, 3.5, 2]


LB=[530, 1, 265.0, 265.0, 1, 100, 10, 0.2, 0.2, 0.5, 10, 10, 0.01, 10, 10, 0.001, 0, 0, 0, 1, 1, 0.01, 1, 1, 1, 0.01, 0, 0, 0.01, 1e-06, 0.1, 0.001, 5, 5, 5, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 100, 6, 6, 10, 0.1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2]

UB = [2650, 530, 2650, 1060, 108, 10000, 1000, 5, 5, 2, 10000, 10000, 100, 1000, 100, 10, 100, 100, 10000, 10000, 10000, 1000, 10000, 10000, 10000, 500, 500, 500, 10000, 1000, 100, 100, 10000, 10000, 10000, 100, 10, 50, 1000, 1000, 1000, 50, 200000, 8, 8, 10000, 50, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 100000, 100000, 100000, 1000000, 2, 300, 2, 1000, 1000, 10000, 10000, 10000, 1000, 10000, 10000, 10000, 125, 10000, 10000, 10000, 100, 10000, 10000, 10000, 35, 10000, 10000, 10000, 100, 1000, 1000, 1000, 100, 1000, 1000, 1000, 15, 5, 10]

# Parameters = [6, 0.666, 20, 0.2, 0.8, 0.5, 3, 0.03, 0.4, 0.08, 0.62, 0.04, 0.8, 1, 0.4, 0.24, 0.086, 0.08, 0.06, 20, 0.21, 0.95, 0.21, 0.8, 0.07, 1.48, 0.66, 150, 20, 2, 1, 2, 20, 20, 530, 54, 7, 90, 18.3, 168.75, 112.5, 12.5, 0, 0, 11.7, 0.63, 1.251, 0, 0, 0, 274, 410, 0.0004, 0.001, 0.0025, 2000, 125, 35, 15, 455.80568000000005, 8000.0, 10000.0, 39860, 4.055722576815053, 18.980999999999998, 18.980999999999998, 258.75, 125.0, 383.75, 174.8310810810811, 193.1310810810811, 185.625]


Variables = 115

def variables(x):
    Q = x[0]
    Qw = x[1]
    Qr = x[2]
    Qef = x[3]
    Qr_p = x[4]
    V_a = x[5]  # Inteira
    A_s = x[6]  # Inteira
    h3 = x[7]
    h4 = x[8]
    r_p = x[9]
    X_I = x[10]
    X_Ir = x[11]
    X_Ief = x[12]
    S_Sent = x[13]
    S_S = x[14]
    S_Oent = x[15]
    S_NOent = x[16]
    S_NO = x[17]
    X_BHent = x[18]
    X_BH = x[19]
    X_BHr = x[20]
    X_BHef = x[21]
    X_Sent = x[22]
    X_S = x[23]
    X_Sr = x[24]
    X_Sef = x[25]
    X_BAent = x[26]
    X_BA = x[27]
    X_BAr = x[28]
    X_BAef = x[29]
    S_NHent = x[30]
    S_NH = x[31]
    X_Pent = x[32]
    X_P = x[33]
    X_Pr = x[34]
    X_Pef = x[35]
    S_NDent = x[36]
    S_ND = x[37]
    X_NDent = x[38]
    X_ND = x[39]
    X_NDr = x[40]
    X_NDef = x[41]
    G_s = x[42]
    S_alkent = x[43]
    S_alk = x[44]
    SSI = x[45]
    SSIef = x[46]
    SSIr = x[47]
    ST = [x[48], x[49], x[50], x[51], x[52], x[53], x[54], x[55], x[56], x[57]]
    v_dn = x[58]
    v_up = x[59]
    v_s = [x[60], x[61], x[62], x[63], x[64], x[65], x[66], x[67], x[68], x[69]]
    J = [x[70], x[71], x[72], x[73], x[74], x[75], x[76], x[77], x[78], x[79]]
    HRT = x[80]
    KLa = x[81]
    r = x[82]
    Sent = x[83]
    S = x[84]
    Xent = x[85]
    X = x[86]
    Xr = x[87]
    Xef = x[88]
    CODent = x[89]
    COD = x[90]
    CODr = x[91]
    CODef = x[92]
    VSSent = x[93]
    VSS = x[94]
    VSSr = x[95]
    VSSef = x[96]
    TSSent = x[97]
    TSS = x[98]
    TSSr = x[99]
    TSSef = x[100]
    BODent = x[101]
    BOD = x[102]
    BODr = x[103]
    BODef = x[104]
    TKNent = x[105]
    TKN = x[106]
    TKNr = x[107]
    TKNef = x[108]
    Nent = x[109]
    N = x[110]
    Nr = x[111]
    Nef	= x[112]
    h = x[113]
    S_O = x[114]

    return Q, Qw, Qr, Qef, Qr_p, V_a, A_s, h3, h4, r_p, X_I, X_Ir, X_Ief, S_Sent, S_S, S_Oent, S_NOent, S_NO, X_BHent, X_BH, X_BHr, X_BHef, X_Sent, X_S, X_Sr, X_Sef, X_BAent, X_BA, X_BAr, X_BAef, S_NHent, S_NH, X_Pent, X_P, X_Pr, X_Pef, S_NDent, S_ND, X_NDent, X_ND, X_NDr, X_NDef, G_s, S_alkent, S_alk, SSI, SSIef, SSIr, ST, v_dn, v_up, v_s, J, HRT, KLa, r, Sent, S, Xent, X, Xr, Xef, CODent, COD, CODr, CODef, VSSent, VSS, VSSr, VSSef, TSSent, TSS, TSSr, TSSef, BODent, BOD, BODr, BODef, TKNent, TKN, TKNr, TKNef, Nent, N, Nr, Nef, h, S_O

def cost_function(x):
    V_a, G_s, A_s, h = x[5], x[42], x[6], x[113]
    f = 174.2214*V_a**1.0699 + 12486.713*G_s**0.6216 + 114.8094*G_s + 955.5*A_s**0.9633 + 41.2706*(A_s*h)**1.0699
    return f
# print(cost_function(v))
def constr_function(x):
    Q, Qw, Qr, Qef, Qr_p, V_a, A_s, h3, h4, r_p, X_I, X_Ir, X_Ief, S_Sent, S_S, S_Oent, S_NOent, S_NO, X_BHent, X_BH, X_BHr, X_BHef, X_Sent, X_S, X_Sr, X_Sef, X_BAent, X_BA, X_BAr, X_BAef, S_NHent, S_NH, X_Pent, X_P, X_Pr, X_Pef, S_NDent, S_ND, X_NDent, X_ND, X_NDr, X_NDef, G_s, S_alkent, S_alk, SSI, SSIef, SSIr, ST, v_dn, v_up, v_s, J, HRT, KLa, r, Sent, S, Xent, X, Xr, Xef, CODent, COD, CODr, CODef, VSSent, VSS, VSSr, VSSef, TSSent, TSS, TSSr, TSSef, BODent, BOD, BODr, BODef, TKNent, TKN, TKNr, TKNef, Nent, N, Nr, Nef, h, S_O = variables(x)

    u_H, Y_H, K_S, K_OH, eta_g, K_NO, k_h, K_X, eta_h, f_p, b_H, b_A, u_A, K_NH, K_OA, Y_A, i_XB, k_a, i_XP, T, P_O2, beta, fracO2, alfa, eta, icv, f_BOD, IVLD, SRT, beta_TSS, beta_COD, beta_BOD, beta_TKN, beta_NO, Qinf, Q_P, S_alkinf, X_Iinf, X_IIinf, X_Sinf, S_Sinf, S_I, S_Oinf, S_NOinf, S_NHinf, S_NDinf, X_NDinf, X_BHinf, X_BAinf, X_Pinf, v1_0, v_0, r_h, f_ns, r_P, ST_t, COD_law, TSS_law, N_law, dens, TSSr_max, TSSr_max_p, Henry, SOST, TKNinf, Ninf, Xinf, Sinf, CODinf, VSSinf, TSSinf, BODinf = list((parameters(read_csv('D:/Mestrado/2ano/Tese/code/param.csv'))).values())

    c = np.empty(1)
    c[0] =Q_P-2*A_s

    ceq = np.empty(99)
    ceq[0]= HRT*Q-V_a
    ceq[1] = -u_H/Y_H*S_S/(K_S+S_S)*(S_O/(K_OH+S_O)+eta_g*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_BH+k_h*X_BH/(K_X*X_BH+X_S)*(S_O/(K_OH+S_O)+eta_h*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_S+Q/V_a*(S_Sent-S_S)
    ceq[2]=(1-f_p)*b_H*X_BH+(1-f_p)*b_A*X_BA-k_h*X_BH/(K_X*X_BH+X_S)*(S_O/(K_OH+S_O)+eta_h*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_S+Q/V_a*(X_Sent-X_S)
    # active heterotrophic biomass – X_BH
    ceq[3] = u_H*S_S/(K_S+S_S)*(S_O/(K_OH+S_O)+eta_g*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_BH-b_H*X_BH+Q/V_a*(X_BHent-X_BH)

    # active autotrophic biomass – X_BA
    ceq[4] = u_A*S_NH/(K_NH+S_NH)*S_O/(K_OA+S_O)*X_BA-b_A*X_BA+Q/V_a*(X_BAent-X_BA)

    # unbiodegradable particulates from cell decay - X_P
    ceq[5] = f_p*b_H*X_BH+f_p*b_A*X_BA+Q/V_a*(X_Pent-X_P)

    # nitrite and nitrate nitrogen – S_NO
    ceq[6] = -(1-Y_H)/(2.86*Y_H)*u_H*S_S/(K_S+S_S)*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO)*eta_g*X_BH+u_A/Y_A*S_NH/(K_NH+S_NH)*S_O/(K_OA+S_O)*X_BA+Q/V_a*(S_NOent-S_NO)

    # ammonia nitrogen - S_NH
    ceq[7] = -u_H*S_S/(K_S+S_S)*(S_O/(K_OH+S_O)+eta_g*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*(i_XB-1)*X_BH-u_A*(i_XB+1/Y_A)*S_NH/(K_NH+S_NH)*S_O/(K_OA+S_O)*X_BA+k_a*S_ND*X_BH+Q/V_a*(S_NHent-S_NH)

    # soluble biodegradable organic nitrogen - S_ND
    ceq[8] = -k_a*X_BH*S_ND+k_h*X_BH/(K_X*X_BH+X_S)*(S_O/(K_OH+S_O)+eta_h*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_ND+Q/V_a*(S_NDent-S_ND)

    # particulate biodegradable organic nitrogen - X_ND
    ceq[9] = b_H*(i_XB-f_p*(i_XP-1))*X_BH+b_A*(i_XB-f_p*(i_XP-1))*X_BA-k_h*X_BH/(K_X*X_BH+X_S)*(S_O/(K_OH+S_O)+eta_h*K_OH/(K_OH+S_O)*S_NO/(K_NO+S_NO))*X_ND+Q/V_a*(X_NDent-X_ND)

    # alklinity – S_alk
    ceq[10] = -i_XB/14*u_H*(S_S/(K_S+S_S))*(S_O/(K_OH+S_O))*X_BH-((1-Y_H)/(14*2.86*Y_H)+i_XB/14)*u_H*(S_S/(K_S+S_S))*(K_OH/(K_OH+S_O))*(S_NO/(K_NO+S_NO))*eta_g*X_BH-(i_XB/14+1/(7*Y_A))*u_A*S_NH/(K_NH+S_NH)*S_O/(K_OA+S_O)*X_BA

     # oxygen balance
    ceq[11] = KLa - alfa*G_s*fracO2*eta*1333.3/(V_a*SOST)*1.024**(T-20)
    ceq[12] = Q*S_Oent - Q*S_O + KLa*(SOST - S_O)*V_a - ((1-Y_H)/Y_H*u_H*(S_S/(K_S+S_S))*(S_O/(K_OH+S_O))*X_BH - (4.57-Y_A)/Y_A*u_A*(S_NH/(K_NH+S_NH))*(S_O/(K_OA+S_O))*X_BA)*V_a

    # composite variables
    # soluble COD
    ceq[13] = Sent - (S_I + S_Sent)
    ceq[14] = S - (S_I + S_S)

    # particulate COD
    ceq[15] = Xent - (X_I + X_Sent + X_BHent + X_BAent + X_Pent)
    ceq[16] = X - (X_I + X_S + X_BH + X_BA + X_P)
    ceq[17] = Xr - (X_Ir + X_Sr + X_BHr + X_BAr + X_Pr)
    ceq[18] = Xef - (X_Ief + X_Sef + X_BHef + X_BAef + X_Pef)

    # total COD
    ceq[19] = CODent - (Xent + Sent)
    ceq[20] = COD - (X + S)
    ceq[21] = CODr - (Xr + S)
    ceq[22] = CODef - (Xef + S)

    # volatile suspended solids
    ceq[23] = VSSent - Xent*1/icv
    ceq[24] = VSS - X*1/icv
    ceq[25] = VSSr - Xr*1/icv
    ceq[26] = VSSef - Xef*1/icv

    # total suspended solids
    ceq[27] = TSSent - (VSSent + SSI)
    ceq[28] = TSS - (VSS + SSI)
    ceq[29] = TSSr - (VSSr + SSIr)
    ceq[30] = TSSef - (VSSef + SSIef)

    # BOD5
    ceq[31] = BODent - f_BOD * (S_Sent + X_Sent + X_BHent + X_BAent)
    ceq[32] = BOD - f_BOD * (S_S + X_S + X_BH + X_BA)
    ceq[33] = BODr - f_BOD * (S_S + X_Sr + X_BHr + X_BAr)
    ceq[34] = BODef - f_BOD * (S_S + X_Sef + X_BHef + X_BAef)

    # TKN notrogen
    ceq[35] = TKNent - (S_NHent + S_NDent + X_NDent + i_XB * (X_BHent + X_BAent) + i_XP * (X_Pent + X_I))
    ceq[36] = TKN - (S_NH + S_ND + X_ND + i_XB * (X_BH + X_BA) + i_XP * (X_P + X_I))
    ceq[37] = TKNr - (S_NH + S_ND + X_NDr + i_XB * (X_BHr + X_BAr) + i_XP * (X_Pr + X_Ir))
    ceq[38] = TKNef - (S_NH + S_ND + X_NDef + i_XB * (X_BHef + X_BAef) + i_XP * (X_Pef + X_Ief))

    # total nitrogen
    ceq[39] = Nent - (TKNent + S_NOent)
    ceq[40] = N - (TKN + S_NO)
    ceq[41] = Nr - (TKNr + S_NO)
    ceq[42] = Nef - (TKNef + S_NO)

    # definitions
    ceq[43] = (Qw * Xr) * SRT - V_a * X

    # suspended matter balances
    ceq[44] = (1 + r) * Qinf * Xent - (Qinf * Xinf + (1 + r) * Qinf * X - V_a * X / (SRT * Xr) * (Xr - Xef) - Qinf * Xef)
    ceq[45] = (1 + r) * Qinf * SSI - (Qinf * TSSinf * 0.2 + (1 + r) * Qinf * SSI - V_a * SSI / (SRT * Xr) * (SSIr - SSIef) - Qinf * SSIef)
    ceq[46] = (1 + r) * Qinf * X_NDent - (Qinf * X_NDinf + (1 + r) * Qinf * X_ND - V_a * X / (SRT * Xr) * (X_NDr - X_NDef) - Qinf * X_NDef)

    # %% dissolved matter balances


    ceq[47] = (1+r)*Qinf*S_Sent - (Qinf*S_Sinf + r*Qinf*S_S)
    ceq[48] = (1+r)*Qinf*S_Oent - (Qinf*S_Oinf + r*Qinf*S_O)
    ceq[49] = (1+r)*Qinf*S_NOent - (Qinf*S_NOinf + r*Qinf*S_NO)
    ceq[50] = (1+r)*Qinf*S_NHent - (Qinf*S_NHinf + r*Qinf*S_NH)
    ceq[51] = (1+r)*Qinf*S_NDent - (Qinf*S_NDinf + r*Qinf*S_ND)
    ceq[52] = (1+r)*Qinf*S_alkent - (Qinf*S_alkinf + r*Qinf*S_alk)


    # flow balances


    ceq[53] = r*Qinf - Qr
    ceq[54] = Q - (Qinf + Qr)
    ceq[55] = Q - (Qef + Qr + Qw)


    # secondary settler ATV model


    ceq[56] = Q_P - 2400*A_s*(0.7*TSS/1000*IVLD)**(-1.34)
    ceq[57] = h3 - 0.3*TSS/1000*V_a*IVLD/(480*A_s)
    ceq[58] = h4 - 0.7*TSS/1000*IVLD/1000
    ceq[59] = r*(TSSr_max-TSS) - TSS
    ceq[60] = r_p*(TSSr_max_p-0.7*TSS) - 0.7*TSS
    ceq[61] = Qr_p - r_p*Q_P
    ceq[62] = VSS - 0.7*TSS
    ceq[63] = VSSef - 0.7*TSSef


    # secondary settler double exponential model


    ceq[64] = v_up*A_s - Qef
    ceq[65] = v_dn*A_s - (Qr + Qw)

    ceq[66] = v_s[0] - np.maximum(0, np.minimum(v1_0, v_0*(np.exp(-r_h*(ST[0]-f_ns*TSS))-np.exp(-r_P*(ST[0]-f_ns*TSS)))))
    ceq[67] = v_s[1] - np.maximum(0, np.minimum(v1_0, v_0*(np.exp(-r_h*(ST[1]-f_ns*TSS))-np.exp(-r_P*(ST[1]-f_ns*TSS)))))
    ceq[68] = v_s[2] - np.maximum(0, np.minimum(v1_0, v_0*(np.exp(-r_h*(ST[2]-f_ns*TSS))-np.exp(-r_P*(ST[2]-f_ns*TSS)))))
    ceq[69] = v_s[3] - np.maximum(0, np.minimum(v1_0, v_0*(np.exp(-r_h*(ST[3]-f_ns*TSS))-np.exp(-r_P*(ST[3]-f_ns*TSS)))))
    ceq[70] = v_s[4] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[4] - f_ns * TSS)) - np.exp(-r_P * (ST[4] - f_ns * TSS)))))
    ceq[71] = v_s[5] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[5] - f_ns * TSS)) - np.exp(-r_P * (ST[5] - f_ns * TSS)))))
    ceq[72] = v_s[6] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[6] - f_ns * TSS)) - np.exp(-r_P * (ST[6] - f_ns * TSS)))))
    ceq[73] = v_s[7] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[7] - f_ns * TSS)) - np.exp(-r_P * (ST[7] - f_ns * TSS)))))
    ceq[74] = v_s[8] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[8] - f_ns * TSS)) - np.exp(-r_P * (ST[8] - f_ns * TSS)))))
    ceq[75] = v_s[9] - np.maximum(0, np.minimum(v1_0, v_0 * (np.exp(-r_h * (ST[9] - f_ns * TSS)) - np.exp(-r_P * (ST[9] - f_ns * TSS)))))

    for j in range(7):
        if ST[j+1] <= ST_t:
            ceq[76+j] = J[j] - v_s[j] * ST[j]
        else:
            ceq[76+j] = J[j] - np.minimum(v_s[j]*ST[j], v_s[j+1]*ST[j+1])

    for j in range(6, 10):
        ceq[76+j] = J[j] - v_s[j] * ST[j]

    # feed layer (m=6)
    j = 6
    ceq[86] = ((Q*TSS)/A_s + J[j-1] - (v_up + v_dn)*ST[j] - np.minimum(J[j], J[j+1])) / (h/10)

    # intermediate layers below the feed layer (m=7 e m=8)
    j = 7
    ceq[87] = (v_dn*(ST[j-1]-ST[j]) + np.minimum(J[j], J[j-1]) - np.minimum(J[j], J[j+1])) / (h/10)
    j = 8
    ceq[88] = (v_dn*(ST[j-1]-ST[j]) + np.minimum(J[j], J[j-1]) - np.minimum(J[j], J[j+1])) / (h/10)

    # lower layer (m=9)
    j = 9
    ceq[89] = (v_dn*(ST[j-1]-ST[j]) + np.minimum(J[j-1], J[j])) / (h/10)

    # intermediate layers above the feed layer (m=1 a 5)
    for j in range(1, 6):
        ceq[89+j] = (v_up*(ST[j+1]-ST[j]) + J[j-1] - J[j]) / (h/10)

    # upper layer (m=0)
    j = 0
    ceq[95] = (v_up*(ST[j+1]-ST[j]) - J[j]) / (h/10)

    ceq[96] = ST[0] - TSSef

    ceq[97] = ST[9] - TSSr

    ceq[98] = h - (h3 + h4 + 1)

    return c, ceq

from HGPSAL.HGPSAL.AUX_Class.Problem_C import Problem
myProblem1 = Problem(Variables, cost_function, LB, UB, constr_function, x0=x0, Variables_C = 1, Variables_Ceq = 99)
opt= {'maxit': 100, 'max_objfun': 2000}
MEGA=(HGPSAL(myProblem1, opt))
print(MEGA)
print(MEGA[5])