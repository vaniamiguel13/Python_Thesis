import math

def objectives(x):
    # Assign variables
    V_a = x[0]
    G_s = x[1]
    A_s = x[2]
    h = x[3]
    Qef = x[4]
    TSSef = x[5]
    CODef = x[6]
    BODef = x[7]
    TKNef = x[8]
    S_NO = x[9]

    # Minimize costs
    f1 = 174.2214 * math.pow(V_a, 1.0699) + 12486.713 * math.pow(G_s, 0.6216) + 114.8094 * G_s + 955.5 * math.pow(A_s,
                                                                                                                  0.9633) + 41.2706 * math.pow(
        A_s * h, 1.0699)

    # Minimize QI
    beta_TSS = 0.8
    beta_COD = 0.2
    beta_BOD = 0.0
    beta_TKN = 0.0
    beta_NO = 0.0
    f2 = 1 / 1000 * (beta_TSS * TSSef + beta_COD * CODef + beta_BOD * BODef + beta_TKN * TKNef + beta_NO * S_NO) * Qef

    return [f1, f2]
