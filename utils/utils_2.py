import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import scipy.integrate as integrate

from scipy.optimize import fsolve
from scipy.stats import betabinom
from scipy.stats import binom


class CF_BL:
    
    def __init__(self, K, p, c,t, rf, M_e, N_e, M_b, N_b, theta_e, theta_b, precision_e, precision_b):
        self.K = K          # Inventory
        self.p = p          # Price
        self.c = c          # Cost 
        self.t = t
        self.rf = rf        # Risk-free rate
        self.M_e = M_e      # Entrepreneur's maximum belief
        self.N_e = N_e      # Entrepreneur's belief dimension
        self.M_b = M_b      # Bank's maximum belief
        self.N_b = N_b      # Bank's belief dimension
        self.theta_e = theta_e  # Entrepreneur's belief parameter
        self.theta_b = theta_b  # Bank's belief parameter

        self.precision_e = precision_e
        self.precision_b = precision_b

        self.a_e = self.precision_e * self.theta_e
        self.b_e = self.precision_e * (1-self.theta_e)

        self.a_b = self.precision_b * self.theta_b
        self.b_b = self.precision_b * (1-self.theta_b)
        
        # Entrepreneur's belief
        self.probability_e_m = lambda m: betabinom.pmf(m, self.M_e, self.a_e, self.b_e)
        
        self.probability_e_D = lambda x, m: betabinom.pmf(x, self.N_e, self.a_e + m, self.b_e + self.M_e - m )
        
        # Bank's belief
        self.probability_b_m = lambda m: betabinom.pmf(m, self.M_b, self.a_b, self.b_b)
        self.probability_b_D = lambda x, m: betabinom.pmf(x, self.N_b, self.a_b + m, self.b_b + self.M_b - m)

        
    
    def spot_market_revenue_knowing_m(self, m):
        """Calculate the spot market revenue given m."""
        integrand = lambda x: np.minimum(x, self.K) * self.probability_e_D(x, m)
        return np.sum(integrand(np.arange(0, self.N_e + 1))) * self.p +   m * self.p * (1-self.t)

    def profit_banque(self, r, m):
        """Calculate the bank's profit given interest rate r and m."""
        S = self.c * (self.K + m) - self.p * (1-self.t) * m
        integrand_b = lambda x: np.minimum(S * (1 + r), np.minimum(x * self.p, self.K* self.p)) * self.probability_b_D(x, m)
        return np.sum(integrand_b(np.arange(0, self.N_b + 1))) - S


    def interest_rate(self, m):
        """Solve for the interest rate that satisfies the bank's profit condition with caching"""
        
        S = self.c * (self.K + m) - self.p* (1-self.t) * m
        
        # Early exit if no loan needed
        if S <= 0:
            return 0
        
        # Define equation for finding interest rate
        equation = lambda r: self.profit_banque(r, m) - S * self.rf
        
        # Solve for interest rate
        try:
            solution = fsolve(equation, x0=0.1, xtol=1e-4)
            
            # Verify solution quality
            if abs(equation(solution[0])) > 1e-2:
                result = np.nan
            else:
                result = max(0, solution[0])  # Ensure non-negative rates
        except:
            result = np.nan
            
        return result

    def CF(self, m):
        """Calculate the cash flow for a given m."""
        production_cost = self.c * (self.K + m)
        
        if production_cost < self.p*(1-self.t) * m:
            return self.spot_market_revenue_knowing_m(m) - production_cost
        
        r = self.interest_rate(m)
        
        if np.isnan(r):
            return 0
        else:
        
            return np.maximum(self.spot_market_revenue_knowing_m(m) - self.c * self.K - (production_cost - self.p* (1-self.t) * m) * r, - self.p* self.t * m)

    def mean_profit(self):
        """Calculate the expected profit of the entrepreneur."""
        integrand_cf = lambda m: self.CF(m) * self.probability_e_m(m)
        return np.sum([integrand_cf(m) for m in range(0, self.M_e + 1)])

    def mean_interest(self):
        """Calculate the expected interest rate."""
        integrand = lambda m: self.interest_rate(m) * self.probability_e_m(m)
        return np.nansum([integrand(m) for m in range(0, self.M_e + 1)])


class BL:
    
    def __init__(self, K, p, c, rf, M_e, N_e, M_b, N_b, theta_e, theta_b, precision_e, precision_b):
        self.K = K          # Inventory
        self.p = p          # Price
        self.c = c          # Cost 
        self.rf = rf        # Risk-free rate

        
        self.M_e = M_e      # Entrepreneur's maximum belief in cf
        self.N_e = N_e      # Entrepreneur's maximum belief in spot market
        
        self.M_b = M_b      # Bank's maximum belief in cf
        self.N_b = N_b      # Bank's maximum belief in spot market

        
        self.theta_e = theta_e  # Entrepreneur's belief parameter
        self.theta_b = theta_b  # Bank's belief parameter
        
        # Entrepreneur's belief

        self.precision_e = precision_e
        self.precision_b = precision_b

        self.a_e = self.precision_e * self.theta_e
        self.b_e = self.precision_e * (1-self.theta_e)

        self.a_b = self.precision_b * self.theta_b
        self.b_b = self.precision_b * (1-self.theta_b)
        
        # Entrepreneur's belief
        self.probability_e = lambda m: betabinom.pmf(m, self.M_e + self.N_e, self.a_e, self.b_e)
 
        
        # Bank's belief
        self.probability_b = lambda m: betabinom.pmf(m, self.M_b + self.N_b , self.a_b, self.b_b)


 
    
    def total_market_revenue(self):
        """Calculate total market revenue."""

        support_x = np.arange(0, self.M_e + self.N_e + 1)

        integrand = lambda x: np.minimum(x, self.K)*self.probability_e(x)

        return np.sum(integrand(support_x)) * self.p
        



    def profit_banque(self, r):
        """Calculate the bank's profit given interest rate r."""

        support_x = np.arange(0, self.M_b + self.N_b + 1)



        integrand_b = lambda x : np.minimum(self.c * self.K * (1 + r), np.minimum(x * self.p, self.K * self.p)) * self.probability_b(x)


  
        return  np.sum(integrand_b(support_x)) - self.c * self.K
        
    def interest_rate(self):
        """Solve for the interest rate that satisfies the bank's profit condition."""
        equation = lambda r: self.profit_banque(r) - self.c * self.K * self.rf
        solution = fsolve(equation, x0=0.1, xtol=1e-4)

        # Return NaN if the solution does not meet tolerance
        if np.abs(equation(solution[0])) > 1e-2:
            return np.nan
        return solution[0]

    

    def mean_profit(self):
        """Calculate the expected profit."""
        r = self.interest_rate()
        if np.isnan(r):
            return 0
        return self.total_market_revenue() - self.c * self.K * (1 + r)

    def mean_interest(self):
        """Return the expected interest rate."""
        return self.interest_rate()


