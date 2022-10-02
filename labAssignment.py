import numpy as np
from scipy.constants import pi, c, k, convert_temperature, e
from scipy.optimize import fsolve, newton_krylov
import matplotlib.pyplot as plt
import pandas as pd



# import dataset
df = pd.read_csv("./I-V-curve_T3.txt", skiprows = 17, encoding="ISO-8859–1", delimiter = "\t")
V_oc = 0.58
I_sc = 4.694
V_mp = 0.47
I_mp = 4.348
T = convert_temperature(22.9, 'Celsius', 'Kelvin')


dIdV = np.diff(df["I in A"])/np.diff(df["V in V"])
D_oc = dIdV[len(dIdV)-1]
D_sc = dIdV[0]



# solve equation

def solve_function(unsolved_value):
    I_L,I_0, a,R_S,R_SH = unsolved_value[0],unsolved_value[1],unsolved_value[2],unsolved_value[3],unsolved_value[4]
    return [
                    I_L - I_0 * (np.exp(e * V_oc/(a * k * T)) - 1) -  V_oc/R_SH,

                    I_sc -  I_L + I_0 * (np.exp((e * I_sc * R_S)/(a* k * T))-1) + (I_sc * R_S)/R_SH,

                    D_oc + I_0 * e / (a * k * T) * (1 + D_oc * R_S) * np.exp(e * V_oc / (a * k * T)) + 1 / R_SH * (1 + D_oc * R_S),

                    D_sc + I_0 * e / (a * k * T) * (1 + D_sc * R_S) * np.exp(e * I_sc * R_S / (a * k * T)) + 1 / R_SH * (1 + D_sc * R_S),

                    -I_mp / V_mp + I_0 * e / (a * k * T) * (1 - I_mp / V_mp * R_S) * np.exp(e * (V_mp + I_mp * R_S)/(a * k * T)) + (1 / R_SH) * (1 - (I_mp / V_mp) * R_S)
                    ]

# reiterate
solved=newton_krylov(solve_function,[I_sc, 0.01, 1, 0.003, 200], iter = 500, rdiff = 0.0001)
#print(newton_krylov(solve_function,[I_sc, 0.01, 1, 0.003, 200], iter = 500, rdiff = 0.0001))

# Reset function
def new_function(I, V):
    I_L,I_0, a,R_S,R_SH = (i for i in solved)
    return I_L - I_0 * (np.exp(e * (V + I * R_S)/(a * k * T)) - 1) - (V + I * R_S) / R_SH - I

# results compare
def draw_curve():
    v_new = np.linspace(0, df["V in V"].max(),1000)
    i = []
    for v in v_new:
        i.append(fsolve(new_function,1,v))
    i_new = np.array(i).flatten()
    plt.plot(v_new, i_new, "b",label="calculate")
    plt.plot(df["V in V"], df["I in A"],"r",label="experimental")
    plt.xlabel("Voltage[V]")
    plt.ylabel("Current[A]")
    plt.legend()
    plt.show()
draw_curve()

