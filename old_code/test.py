
from utils import BL, CF_BL, oracle_bl, oracle_CF_BL



N_e,N_b = 100,100 # spot market size
M_e,M_b = 40,40 # crowdfundig market size





K = 60 # inventory
p = 1  # price
c = 0.6 # unit cost of production
t = 0.1
rf = 0.05 # risk free interest


theta_e = 0.5

theta_b=0.5
precision_e = 100
precision_b = 100



print(CF_BL(K, p,c,t,rf, M_e,N_e, M_b, N_b, theta_e, theta_b , precision_e, precision_b).mean_profit())

print(BL(K, p,c,rf, M_e,N_e, M_b, N_b, theta_e, theta_b, precision_e, precision_b ).mean_profit())

print(oracle_bl(K, p,c,rf, M_e,N_e, M_b, N_b, theta_e, theta_b,  precision_b ).mean_profit())

print(oracle_CF_BL(K, p,c, t,rf, M_e,N_e, M_b, N_b, theta_e, theta_b, precision_b ).mean_profit())