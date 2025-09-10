import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.signal.windows import blackmanharris
import matplotlib.pyplot as plt
from matplotlib import cm
from math import ceil 
from tqdm import tqdm
from itertools import chain
import bisect

import os 
import time

from os import makedirs, getcwd
from os.path import exists

import sys

# Import functions from config_reader.py
import config_reader as cr

# Import function from plot_results.py
import plot_results as pr

''' 
Python Script Implementation of the 2-step Lax-Wendroff scheme for solving the 1D NSE in one blood vessel with different boundary conditions.
This new version aims to solve the NSE in a vessel network declared in the config file.

Parameters are being declared in the config before being read by the main program.
The inlet data for either pressure or flow can be read by the program as long as the file's path is mentionned in the config file
The csv file should contain two columns, the first one being the time values and the second one being the flow rate values at the inlet of the 
vessel in centimeters cubed per second.

If you have no data to run the simulation, please leave the path variable empty in the config file and choose pressure as an inlet boundary condition.
The programm will genereate a pulse waveform by itself.

Data example_inlet.csv available from the simulations ran by Kolachalama V, Bressloff NW, Nair PB and Shearman CP (2007) 
Predictive Haemodynamics in a One-Dimensional Carotid Artery Bifurcation. 
Part I: Application to Stent Design. IEEE Transactions on Biomedical Engineering 54 (5): 802-812, doi 10.1109/TBME.2006.889188

To do :
- Split all the functions into different files to make it more readable
- Writing a demonstrator
'''

def CFL_condition(u, a0, alpha, rho, dx, dt, n, nt):
	"""
	Allows us to check if the CFL condition is satisfied
	
	Parameters :
	- u : float, average velocity at a given time step
	- a0 : float, cross sectional area at rest
	- alpha : float, elatance coefficient
	- rho : float, blood density 
	- dx : float, space step
	- dt : float, time step
	- n : int current time step
	- nt : total number of time steps

	Returns :
	- No returned values if CFL respected (True), raise ValueError if not respected (False)
	"""
	
	wave_speed = np.sqrt(alpha*a0/rho)
	if np.max([u + wave_speed, u-wave_speed])*(dt/dx) <= 1:
		if n == 1:
			print(f"Wave speed [cm/s] : {wave_speed:.2e}")
		if (n == nt-1):
			print("CFL Condition was satisfied for every time steps")
		else:
			pass
	else:
		print(f"Wave speed [cm/s] : {wave_speed:.2e}")
		print(f"u : {u:.2e} a0 : {a0:.2e} alpha : {alpha:.2e} rho : {rho:.2e} dt : {dt:.2e} dx : {dx:.2e}")
		raise ValueError("The CFL condition isn't satisfied, the time step needs to be lower")

def blackman_harris_window(t):
	"""
	Function that returns normalized value of the blackman harris window for t in [0,1]
	
	Parameters :
	- t : given time in 0,1
	
	Returns : 
	- w(t) normalized value
	"""
	N = len(t)
	window = blackmanharris(N, sym=False)

	if np.isscalar(t):
		if 0 <= t < 1:
			index = int(t * N)
			return window[index]
		else:
			return 0.0

	result = np.zeros_like(t)
	valid = (t >= 0) & (t < 1)
	indices = (t[valid] * N).astype(int)
	result[valid] = window[indices] # ??? 
	return result

def Three_wave_component(t):
	"""
	Creates a pressure function obtained as a sum of simulated percussion, tidal and dicrotic  wave
	components, with a heart rate of 80 beats per minutes 

	From :  Takumi Nagasawa et al. « Blood Pressure Estimation by Photoplethysmogram
	Decomposition into Hyperbolic Secant Waves ». In : Applied Sciences 12.4 (2022).
	issn : 2076-3417. doi : 10.3390/app12041798.

	Parameters :
	- t : Array corresponding to the time in seconds

	Returns :
	- P(t) : Array corresponding to the blood flow at the inlet for the time t
	"""
	relative_pulse_amplitude = 89	#Relative pulse scaling
	
	# Three-phase pulse parameters (user-defined as fractions of the heart cycle)
	amp_phase1 = 0.5 				# Amplitude of phase 1 (systolic peak)
	amp_phase2 = 0.3 				# Amplitude of phase 2 (diastolic trough)
	amp_phase3 = 0.25 				# Amplitude of phase 3 (resting phase)

	t_phase1 = 0.05 
	t_phase2 = 0.2 
	t_phase3 = 0.38 

	len_phase1 = 0.55
	len_phase2 = 0.55
	len_phase3 = 0.6 

	t_compute = (t%T)/T
	tau_1 = (t_compute  - t_phase1)/((len_phase1))
	tau_2= (t_compute - t_phase2)/((len_phase2))
	tau_3 = (t_compute - t_phase3)/((len_phase3))

	P_res = relative_pulse_amplitude*(amp_phase1*blackman_harris_window(tau_1) + 
			amp_phase2*blackman_harris_window(tau_2) + 
			amp_phase3*blackman_harris_window(tau_3))

	return P_res

def Flow_sim(t):
	"""
	Function trying to simulate an inflow function, does not aim to be realistic
	
	Parameters :
		- t : array containing each time step
	
	Returns : Q(0,t)
	"""
	return 0

def get_data(arg, inlet,t,  **kwargs):
	"""
	Returns the inlet flow rate at time t (either float or array)

	Parameters : 
	- argument passed when calling the python script, the file must contain the data for ONE cycle only
	- inlet : str containing which boundary condition to choose (pressure inlet or inflow)
	- t : array containing the time values for the simulation in seconds

	- Keyword Argument :
		- hr : int corresponding to the number of bpm

	Returns :
	- f : Function interpolating flow rate from the csv file [cm3/s]
	- T : float representing the period length [s]

	TO EDIT : scheme for more adaptibility : 
	1) Check inlet bc
	2) Check if file
	3) Check if heart rate given
	"""

	if arg == 'None' or arg == '':
		if 'hr' in kwargs:
			T = 60/kwargs['hr']
		else:
			raise ValueError("There is no value to the heart rate.")
		
		if inlet == 'pressure':
			return Three_wave_component
		elif inlet == 'flow':
			print("No path folder mentionned, using the inflow from the example")
			plt.figure()
			plt.plot(t,f)
			plt.show()
			return Flow_sim
		else:
			raise ValueError("inlet string in config file must be either pressure or flow")
	else:
		data = pd.read_csv(arg, header=None)
		t_data = data[0].values
		y_data = data[1].values
		f = interp1d(t_data, y_data, kind='linear')
		T = t_data[-1]
		f = f(periodic(t,T))/5.7
		
	return f

def write_data(dict_data, x_plot, time_plot):
	"""
	Function allowing us to export the data generated by the simulation into csv files

	Parameters : 
		- dict_data : dictionnary containing the hemodynamic value for each vessel over one cardiac cycle
		- x_plot : dictionnary of each vessel discretization
		- time_plot : array, time discretization
	"""
	folder_name = 'res_' + time.strftime("%Y_%m_%d_%H_%M_%S")
	path = "data/" + folder_name + "/"
	if not exists(path):
		makedirs(path)
	np.savetxt(path+"time_data.csv", time_plot[:], delimiter=',')
	for data_name, data in dict_data.items():
		if data is not None:
			text = data_name.split("_")
			np.savetxt(path+f"{text[0]}_{text[1]}.csv", data[:,:], delimiter=',')
	
	for x_name, x in x_plot.items():
		if x is not None:
			text = x_name.split("_")
			np.savetxt(path+f"{text[0]}_{text[1]}.csv", x[:], delimiter=',')
	print(80*"-")
	print("Data have been saved in this directory : ", getcwd()+"/"+path)

def periodic(t, T):
	"""
	Returns equivalent time of the first period

	Parameters :
	- t : float representing the current time
	- T : float representing the period length

	Returns :
	- t % T : float representing the equivalent time in the first period
	"""
	return t % T

def inlet_bc(vessel, data_inlet, dt):
	"""
	Applies the inlet boundary condition for the vessel.
	doi 10.1109/TBME.2006.889188 | doi 10.1114/1.1326031

	Parameters :
	- vessel : dict containing all the vessel parameter
	- data_inlet : array containing the flow value for the current time and previous half time step
	- dt : float, time step

	Returns :
	- a_inlet : float corresponding to the cross-sectional area at the inlet of the vessel
	- q_inlet : float corresponding to the flow rate at the inlet of the vessel
	"""

	dtdx = dt/vessel["dx"]
	dtover4 = dt/4

	# Unpacking the vessel parameters
	A = vessel["A"]
	Q = vessel["Q"]
	
	q_inlet = data_inlet[1] # q^{n+1}_0
	q_nph_0 = data_inlet[0] # q^{n+1/2}_0
	q_np_half = 0.5*(Q[1] + Q[0]) - 0.5 * (dtdx) * (Flux(vessel, j = 1, k = 2) - Flux(vessel,j = 0, k = 1))\
			  + dtover4  * (Source(vessel, j = 1, k = 2) + Source(vessel, j = 0, k = 1)) # q^{n+1/2}_{1/2}
	#q_m_half = 2*q_nph_0 - q_np_half # q^{n+1/2}_{-1/2}
	a_inlet = A[0] - (dtdx) * 2 * (q_np_half - q_nph_0) # A^{n+1}_0 Combines richtmeyer scheme for A and ghost point method
	return a_inlet[0], q_inlet

def WK_outlet_bc(vessel, dt, **kwargs):
	"""
	Computes the outlet boundary condition basing on the three element windkessel (3WK) model by using the fixed point iteration method.
	doi 10.1109/TBME.2006.889188 | doi 10.1114/1.1326031
	Parameters : 
	- vessel : dict containing the parameters of the vessel
	- dt : float representing the time step

	Keyword Arguments :
		- diast : diastolic pressure in Pa (default 70 Pa)
		
	Returns :
	- a_outlet : float corresponding to the cross-sectional area at the outlet of the vessel
	- q_outlet : float corresponding to the flow rate at the outlet of the vessel
	"""

	if "diast" in kwargs:
		diastolic_pressure = kwargs["diast"]
	else:
		diastolic_pressure = 70 
	
	# Unpacking the vessel parameters
	R1 = vessel["R1"]
	R2 = vessel["R2"]
	C = vessel["C"]
	Q = vessel["Q"]
	A = vessel["A"]
	A0 = vessel["A0"]
	dx = vessel["dx"]
	mu = vessel["mu"]
	rho = vessel["rho"]
	alpha = vessel["alpha"]
	dalpha_half_p = vessel["dalpha_half_p"]
	dalpha_half_m = vessel["dalpha_half_m"]
	alpha_half_p = vessel["alpha_half_p"]
	alpha_half_m = vessel["alpha_half_m"]
	drdzhp = vessel["drdz_half_p"]
	drdzhm = vessel["drdz_half_m"]
	A0_half_p = vessel["A0_half_p"]
	A0_half_m = vessel["A0_half_m"]

	q_n = Q[-1] 
	a_n = A[-1]
	diast_pa = diastolic_pressure * 1333.22365	# mmHg to g/(cm.s^2)
	if vessel["p_type"] == 'linear':
		p_m_p1 = p_n = alpha[-1]*(a_n - A0[-1]) + diast_pa # Initial guess for pressure at outlet (same as before, i.e p^{n+1}_M = p^n_M)
	else:
		p_m_p1 = p_n = alpha[-1]*(1- np.sqrt(A0[-1]/a_n)) + diast_pa

	A_np_mp = 0.5 * (A[-1] + A[-2]) - 0.5 * (dt/dx) * (Q[-1] - Q[-2])
	A_np_mm = 0.5 * (A[-2] + A[-3]) - 0.5 * (dt/dx) * (Q[-2] - Q[-3])

	Q_np_mp = (0.5 * (Q[-1] + Q[-2]) - 0.5 * (dt/dx) * (Flux(vessel, j = -1) - Flux(vessel, j = -2, k = -1))\
			+ (dt/4) * (Source(vessel, j = -1) + Source(vessel, j = -2, k = -1)))[0]
	
	Q_np_mm = (0.5 * (Q[-2] + Q[-3]) - 0.5 * (dt/dx) * (Flux(vessel, j = -2, k = -1) - Flux(vessel, j = -3, k = -2))\
			+ (dt/4) * (Source(vessel, j = -2, k = -1) + Source(vessel, j = -3, k = -2)))[0]
	
	if vessel["p_type"] == 'linear':
		Q_mm = Q[-2] - dt/dx * (Q_np_mp**2/A_np_mp + alpha_half_p[-2]/(2*rho) * (A_np_mp**2 - A0_half_p[-2]**2) - Q_np_mm**2/A_np_mm - alpha_half_m[-2]/(2*rho) * (A_np_mm**2 - A0_half_m[-2]**2))\
			 + (dt/2) * (-(8*np.pi*mu)/(rho) * (Q_np_mp/A_np_mp + Q_np_mm/A_np_mm) + (1/rho) * ((2*np.pi*(np.sqrt(A_np_mp/np.pi))*alpha_half_p[-2]*(A_np_mp-A0_half_p[-2]) - ((A_np_mp-A0_half_p[-2])**2)*dalpha_half_p[-2])* drdzhp[-2] \
																					   + (2*np.pi*(np.sqrt(A_np_mm/np.pi))*alpha_half_m[-2]*(A_np_mm-A0_half_m[-2]) - ((A_np_mm-A0_half_m[-2])**2)*dalpha_half_m[-2])* drdzhm[-2]))
	else:
		Q_mm = Q[-2] - dt/dx * (Q_np_mp**2/A_np_mp + alpha_half_p[-2]/(rho) * (np.sqrt(A_np_mp*A0_half_p[-2])) - Q_np_mm**2/A_np_mm - alpha_half_m[-2]/(rho) * (np.sqrt(A_np_mm*A0_half_m[-2])))\
			 + (dt/2*rho) * (-(8*np.pi*mu) * (Q_np_mp/A_np_mp + Q_np_mm/A_np_mm) + (2*np.sqrt(A_np_mp)*(np.sqrt(np.pi)*alpha_half_p[-2] + np.sqrt(A0_half_p[-2])*dalpha_half_p[-2])-A_np_mp*dalpha_half_p[-2])*drdzhp[-2] \
					+ (2*np.sqrt(A_np_mm)*(np.sqrt(np.pi)*alpha_half_m[-2] + np.sqrt(A0_half_m[-2])*dalpha_half_m[-2])-A_np_mm*dalpha_half_m[-2])*drdzhm[-2])
	
	k = 0
	while k < 1000:
		p_old = p_m_p1
		q_outlet = ((p_old - p_n)/R1) + q_n + (dt*p_n)/(R1*C*R2) - (q_n*(R1+R2)*dt)/(C*R1*R2)
		a_outlet = a_n - (dt/dx)*(q_outlet - Q_mm)
		if vessel["p_type"] == 'linear':
			p_m_p1 = alpha[-1]*(a_outlet - A0[-1]) + diast_pa #+ diastolic_pressure*1333.22
		else:
			p_m_p1 = alpha[-1]*(1- np.sqrt(A0[-1]/a_outlet)) +diast_pa # + diastolic_pressure*1333.22
		if abs(p_m_p1 - p_old) < 1e-7:
			break
		k +=1
	return a_outlet, q_outlet
 
def pressure_outlet(vessel, data_outlet, dt, **kwargs):
	"""
	Computes the flow and area at the outlet of the domain given a pressure boundary condition

	Parameters :
	- vessel : dict containing the parameters of the vessel
	- data_oulet : array containing pressure data at full and half time step at the outlet
	- dt : time step

	Keyword Arguments :
		- data_outlet : array containing the pressure at the outlet of the vessel for each time step

	Returns : 
	- a_out : value of the cross sectional area at the outlet
	- q_out : value of the blood flow at the outlet of the vessel
	"""

	Q = vessel["Q"]
	A = vessel["A"]
	A0 = vessel["A0"]
	A0_half_p = vessel["A0_half_p"]
	A0_half_m = vessel["A0_half_m"]
	dx = vessel["dx"]
	mu = vessel["mu"]
	rho = vessel["rho"]
	alpha = vessel["alpha"]
	dalpha_half_p = vessel["dalpha_half_p"]
	dalpha_half_m = vessel["dalpha_half_m"]
	alpha_half_p = vessel["alpha_half_p"]
	alpha_half_m = vessel["alpha_half_m"]
	drdz_half_p = vessel["drdz_half_p"]
	drdz_half_m = vessel["drdz_half_m"]

	a_n = A[-1]
	q_n = Q[-1]
	
	a_out = (data_outlet[1]/alpha[-1]) + A0[-1]
	a_out_mm = (data_outlet[0])/alpha[-1] + A0[-1]
	
	a_np_mm = 0.5*(A[-1] + A[-2]) - dt/(2*dx) *(Q[-1] - Q[-2])
	q_np_mm = (0.5 * (Q[-1] + Q[-2]) - 0.5 * (dt/dx) * (Flux(vessel, j = -1) - Flux(vessel, j = -2, k = -1)) \
				+ (dt/2) * (Source(vessel, j = -2, k = -1) + Source(vessel, j = -1))/2)[0] # q^{n+1/2}_{N-1/2}
	
	a_np_mp = 2*a_out - a_out_mm
	q_np_mp = (a_n - a_out)*(dx/dt) + q_np_mm
	
	if vessel["p_type"] == "linear":
		q_out = q_n - (dt/dx) * (q_np_mp**2/a_np_mp + (alpha_half_p[-1]/(2*rho)) * (a_np_mp - A0_half_p[-1]) - (q_np_mm**2/a_np_mm + (alpha_half_m[-1]/(2*rho)) * (a_np_mm - A0_half_m[-1]))) \
		 + (dt/2) * (-(8*np.pi*mu)/(rho) * (q_np_mp/a_np_mp + q_np_mm/a_np_mm) + (1/(2*rho))*((a_np_mp**2 - A0_half_p[-1]**2)*dalpha_half_p[-1] - 4*((np.pi)**2) * (np.sqrt(A0_half_p[-1]/np.pi))**3 * alpha_half_p[-1])*drdz_half_p[-1] \
			+ (1/(2*rho))*((a_np_mm**2 - A0_half_m[-1]**2)*dalpha_half_m[-1] - 4*((np.pi)**2) * (np.sqrt(A0_half_m[-1]/np.pi))**3 * alpha_half_m[-1])*drdz_half_m[-1])
	else:
		q_out = Q[-1] - (dt/dx) * (q_np_mp**2/a_np_mp + alpha_half_p[-1]/(rho) * (np.sqrt(a_np_mp*A0_half_p[-1])) - q_np_mm**2/a_np_mm - alpha_half_m[-1]/(rho) * (np.sqrt(a_np_mm*A0_half_m[-1])))\
			 + (dt/2*rho) * (-(8*np.pi*mu) * (q_np_mp/a_np_mp + q_np_mm/a_np_mm) + (2*np.sqrt(a_np_mp)*(np.sqrt(np.pi)*alpha_half_p[-1] + np.sqrt(A0_half_p[-1])*dalpha_half_p[-1])-a_np_mp)*drdz_half_p[-1] \
					+ (2*np.sqrt(a_np_mm)*(np.sqrt(np.pi)*alpha_half_m[-1] + np.sqrt(A0_half_m[-1])*dalpha_half_m[-1])-a_np_mm)*drdz_half_m[-1])
	return a_out, q_out	

def pressure_inlet(vessel, data_inlet, dt):
	"""
	Computes the flow and cross sectional area at the inlet of the vessel, given a pressure in Pa 
	NB : We suppose the absence of vessel anomaly at the inlet to simplify the computation
	Parameters :
	- vessel : dict containing all the vessel parameter
	- data_inlet : array containing the flow value for the current time and previous half time step
	- dt : float, time step

	Returns : 
	- a_in : value of the cross sectional area at the inlet
	- q_in : value of the blood flow at the inlet of the vessel
	"""
	a_0 = vessel["A"][0]
	q_0 = vessel["Q"][0]

	if vessel["p_type"] == 'linear':
		a_in = ((data_inlet[1]/vessel["alpha"][0]))+vessel["A0"][0]
		a_0_mm = data_inlet[0]/vessel["alpha"][0] + vessel["A0"][0]
	else:
		a_in = (vessel["A0"][0]*vessel["alpha"][0]**2)/((vessel["alpha"][0] - data_inlet[1])**2)
		a_0_mm = (vessel["A0"][0]*vessel["alpha"][0]**2)/((vessel["alpha"][0] - data_inlet[0])**2)

	A = vessel["A"]
	Q = vessel["Q"]
	dx = vessel["dx"]

	vessel["A_half_p"][0] = 0.5*(A[0] + A[1]) - dt/(2*dx) * (Q[1] - Q[0])
	vessel["Q_half_p"][0] = (0.5*(Q[1] + Q[0]) - 0.5 * (dt/dx) * (Flux(vessel, j = 1, k = 2) - Flux(vessel,j = 0, k = 1))\
			  + dt/2 * (Source(vessel, j = 1, k = 2) + Source(vessel, j = 0, k = 1))/2)[0] # q^{n+1/2}_{1/2}
	
	vessel["A_half_m"][0] = 2*a_0_mm - vessel["A_half_p"][0]
	vessel["Q_half_m"][0] = vessel["Q_half_p"][0] + (dx/dt) * (a_in - a_0)

	q_in = q_0 - (dt/dx) * ((Flux(vessel, j = 0, k = 1, p = 1)- Flux(vessel, j = 0, k = 1, p = 0)) + dt/2 * (Source(vessel, j = 0, k = 1, p = 0) + Source(vessel, j = 0, k = 1, p = 1)))[0]
		
	return a_in, q_in

def HicksHenneBump(x, x_start, x_end, x_anomaly, x_width, a, r0, stenosis, aneurysm):
	"""
	Returns the cross-sectional area of the vessel with a bump [needs update to generate a function which is C1]

	Parameters :
	- x : float representing the spatial point (for convenience we normalize to get [0,1])
	- x_anomaly : float representing the location of the maximum point of the bump (in [0,1])
	- x_width : float representing the width of the bump (in [0,L])
	- a : float representing the amplitude of the bump (in [0,r0])
	- r0 : float representing the initial radius of the vessel

	Returns :
	- r : float representing the radius at the spatial point x
	"""
	r = np.full_like(x, r0)
	xi = x/np.max(x)
	x_anom = x_anomaly / np.max(x)
	print(x_anom)
	if stenosis:
		r -= a*(np.sin(np.pi*xi**(np.log(0.5)/np.log(x_anom)))**(x_width))
	elif aneurysm:
		r += a*(np.sin(np.pi*xi**(np.log(0.5)/np.log(x_anom)))**(x_width))
	return r

def dHdx(x, x_start, x_end, x_anomaly, x_width, a, stenosis, aneurysm):
	res = np.zeros_like(x)
	x_anom = x_anomaly / np.max(x)
	exponent = np.log(0.5)/np.log(x_anom)
	xi = x/np.max(x)
	if stenosis or aneurysm:
		f = xi ** exponent
		df_dxi = exponent * xi ** (exponent - 1)
		sinf = np.sin(np.pi * f)
		cosf = np.cos(np.pi * f)
		dxi_dx = 1 / (x_end - x_start)
		res = a * x_width * sinf ** (x_width - 1) * np.pi * cosf * df_dxi * dxi_dx
		if stenosis:
			res = res * -1.0
	return res

def radius(Rtop, Rbottom, x, L):
	"""
	Computes the vessel radius at rest and its derivative of the vessel by considering the vessel hasn't any anomalies
	"""
	
	r = Rtop*(Rbottom/Rtop)**(x/L)
	drdz = np.log(Rbottom/Rtop)/L * r
	return r, drdz

def drdx(x, stenosis, aneurysm, x_anomaly, x_width, a, x_start, x_end):
	"""
	Placeholder but will be used to compute dr0/dz used for the source term,

	Parameters :
	- x : array containing Discretized artery
	- stenosis, aneurysm : boolean representing if the vessel has an anomaly or not
	- x_anomaly : float representing the location of the maximum point of the bump (in [0,1])
	- x_width : float representing the width of the bump (in [0,L])
	- a : float representing the amplitude of the bump (in [0,r0])
	- x_start : beginning of the anomaly 
	- x_end : end of the anomaly

	Returns :
	- dr : array corresponding to the exact computed value of dr/dz
	"""
	if stenosis or aneurysm:
		x_anonm = x_anomaly / np.max/(x)
		xi = x/np.max(x)
		exponent = np.log(0.5) / np.log(x_anomaly)
		sin_arg = np.pi * xi**exponent
		dr_anomaly = (-np.log(2) * np.pi * a * xi**(-1 + exponent) * x_width * np.cos(sin_arg) * (np.sin(sin_arg) ** (x_width - 1)) / np.log(x_anomaly))
		#dr_anomaly = a*(np.pi**x_width)*powx*(y**(powx-1))*np.cos((np.pi**x_width)*(y**powx))
		
		dr = dr_anomaly
		if aneurysm:
			return dr
		else:
			return -dr
	else:
		return np.zeros_like(x)

def Flux(vessel,  **kwargs):
	"""
	Computes the flux term used during the discretisation

	Parameters : 
	- vessel : dictionary containing the vessel parameters

	- Keywords arguments :
		- j : index variable = start (mandatory)
		- k : index variable = end (leave blank if until last coef)
		- p : index variable : 1 for half step +1/2, 0 for half step -1/2
	"""
	if 'p' in kwargs:
		p = kwargs['p']
		if p == 1:
			q = vessel["Q_half_p"]
			a = vessel["A_half_p"]
			al = vessel["alpha_half_p"]
			a0 = vessel["A0_half_p"]
		else:
			q = vessel["Q_half_m"]
			a = vessel["A_half_m"]
			al = vessel["alpha_half_m"]
			a0 = vessel["A0_half_m"]
	else:
		q = vessel["Q"]
		a = vessel["A"]
		al = vessel["alpha"]
		a0 = vessel["A0"]
	if 'j' in kwargs:
		j = kwargs['j']
		if j < 0:
			j = len(vessel["Q"]) + j
		else:
			pass
		q = q[j:]
		a = a[j:]
		a0 = a0[j:]
		al = al[j:]
	if 'k' in kwargs:
		k = kwargs['k']
		if k < 0:
			k = len(vessel["Q"]) + k
		else:
			pass
		q = q[:k-j]
		a = a[:k-j]
		a0 = a0[:k-j]
		al = al[:k-j]
	if 'j' not in kwargs and 'k' not in kwargs:
		raise IndexError("At least the start index 'j' must be specified")
	rho = vessel["rho"]
	res = np.pow(q,2)/a 
	if vessel["p_type"] == 'linear':
		res += (al/(2*rho))*(np.pow(a,2)-np.pow(a0,2))
	else:
		res += (al/rho)*(np.sqrt(a0*a))
	return res

def Source(vessel, **kwargs):
	"""
	Computes the source term used during the discretisation

	Parameters : 
	- vessel : dictionary containing the vessel parameters

	- Keywords arguments :
		- j : index variable = start (mandatory)
		- k : index variable = end (leave blank if until last coef)
		- p : index variable : 1 for half step +1/2, 0 for half step -1/2
	"""
	if 'p' in kwargs:
		p = kwargs['p']
		if p == 1:
			q = vessel["Q_half_p"]
			a = vessel["A_half_p"]
			al = vessel["alpha_half_p"]
			a0 = vessel["A0_half_p"]
			dal = vessel["dalpha_half_p"]
			drdz = vessel["drdz_half_p"]
		else:
			q = vessel["Q_half_m"]
			a = vessel["A_half_m"]
			al = vessel["alpha_half_m"]
			a0 = vessel["A0_half_m"]
			dal = vessel["dalpha_half_m"]
			drdz = vessel["drdz_half_m"]
	else:
		q = vessel["Q"]
		a = vessel["A"]
		al = vessel["alpha"]
		a0 = vessel["A0"]
		dal = vessel["dalphadr"]
		drdz = vessel["drdz"]

	if 'j' in kwargs:
		j = kwargs['j']
		if j < 0:
			j = len(vessel["Q"]) + j
		else:
			pass
		q = q[j:]
		a = a[j:]
		a0 = a0[j:]
		drdz = drdz[j:]
		al = al[j:]
		dal = dal[j:]
	if 'k' in kwargs:
		k = kwargs['k']
		if k < 0:
			k = len(vessel["Q"]) + k
		else:
			pass
		q = q[:k-j]
		a = a[:k-j]
		a0 = a0[:k-j]
		drdz = drdz[:k-j]
		al = al[:k-j]
		dal = dal[:k-j]
	if 'j' not in kwargs and 'k' not in kwargs:
		raise IndexError("At least the start index 'j' must be specified")
	r0 = np.sqrt(a0/np.pi)
	rho = vessel["rho"]
	mu = vessel["mu"]
	res = -(8*np.pi*mu)/(rho)*(q/a)
	if vessel["p_type"] == 'linear':
		res += (1/rho)*(al*(a-a0))*(2*np.pi*r0 - ((a-a0)/al) * dal)*drdz
	else:
		res += (1/rho)*(2*np.sqrt(a)*(np.sqrt(np.pi)*al + np.sqrt(a0)*dal)-a*dal)*drdz
	return res

def gaussian_aneurysm(x, A, mu, sigma):
    """
    Function that return the radius of a vessel concerned by an aneurism at each point of the discretization.
    Contrary to HicksHenneBump function, this function is C1.
    
    Parameters :
    - x: float/array : spatial points
    - A: float corresponding to the normalizing constant (= amplitude of the bump)
    - mu: location of the amplitude (in [0,L])
    - sigma: Standard deviation  (in ]0, min(mu, abs(mu-L))/3])
    
    Returns: 
	- res : Value of the gaussian function (vessel radius) at the spatial point x
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def gaussian_derivative(x, A, mu, sigma):
    """
    Derivative of the gaussian function which is used to define the vessel radius
    
    Parameters :
    - x: float/array : spatial points
    - A: float corresponding to the normalizing constant (= amplitude of the bump)
    - mu: location of the amplitude (in [0,L])
    - sigma: Standard deviation  (in ]0, min(mu, abs(mu-L))/3])
    
    Returns :
    - res : Value of the derivative at the spatial point x
    """
    return - A * (x - mu) / (sigma ** 2) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def pressure(A, A0,alpha, diast, pressure_type):
	"""
	Computes the blood pressure 
	
	Parameters : 
	- alpha, 1D array containing elastance coefficient 
	- A, array containing cross sectionnal area
	- A0, array for the cross-sectional area at rest

	Returns :
	- P, blood pressure
	"""
	P = np.zeros_like(A)

	if pressure_type == 'linear':
		P = alpha * (A - A0)/1333.22 + diast
	elif pressure_type == 'sqrt':
		P = alpha * (1-np.pow(A0/A, 0.5))/1333.22 + diast
	return P

def al_artery(r, A0, pressure_type,  k1, k2, k3):
	"""
	Computes the elastance coefficient for the artery based on the radius and cross-sectional area at rest
	Parameters to estimate alpha empirically based on https://doi.org/10.1114/1.1326031

	Parameters :
	- r : array containing the radius of the vessel at each spatial point
	- A0 : float representing the cross-sectional area at rest
	- pressure_type : string, correspond to the pressure model applied 
	- k1 : elasticity parameter [g/(s² cm)]
	- k2 : elasticity parameter [cm^-1]
	- k3 : elasticity parameter [g/(s² cm)]

	Returns :
	- alpha : array containing the elastance coefficient for each spatial point
	"""
	al = (k1*np.exp(k2*r) + k3)
	if pressure_type == 'linear':
		al = al/(2*A0)
	return al

def daldr(r, A0, pressure_type,  k1, k2, k3):
	"""
	Computes the derivative of the elastance coefficient with respect to the radius
	Parameters to estimate alpha empirically based on https://doi.org/10.1114/1.1326031

	Parameters :
	- r : array containing the radius of the vessel at each spatial point
	- A0 : float representing the cross-sectional area at rest
	- pressure_type : string, correspond to the pressure model applied 
	- k1 : elasticity parameter [g/(s² cm)]
	- k2 : elasticity parameter [cm^-1]
	- k3 : elasticity parameter [g/(s² cm)]

	Returns :
	- dalphadr : array containing the derivative of the elastance coefficient for each spatial point
	"""
	dal = (k1*k2*np.exp(k2*r))
	if pressure_type == 'linear':
		dal = dal/(2*A0) - (1/(A0*r))*(k1*np.exp(k2*r) + k3)
	return dal

def create_vessel(R1, R2, C, L, ru_0, rd_0, dx, mu, rho, pressure_type, dict_ano, k1, k2, k3):
	"""
	Creates a vessel with a given length and cross-sectional area at rest

	Parameters :
	- R1, R2 : floats corresponding to two resistances [g/s * cm^{-4}]
	- C : float , compliance coefficient [cm^4.s^2/g]
	- L : float representing the length of the vessel [cm]
	- ru_0 : float representing the radius at rest at the inlet [cm]
	- rd_0 : float representing the radius at rest at the outlet [cm]
	- dx : float space discretization [cm]
	- mu : float blood viscosity [g/(cm.s)]
	- rho : float blood density [g/cm^3]
	- pressure_type : string, correspond to the pressure model applied 
	- dict_ano : dictionnary containing anomalies parameters, either stenosis or aneurysm
	- k1 : elasticity parameter [g/(s² cm)]
	- k2 : elasticity parameter [cm^-1]
	- k3 : elasticity parameter [g/(s² cm)]

	Returns :
	- x : array of spatial points
	- A : array of cross-sectional area at rest

	"""
	nx = int(L/dx)+1
	if nx-1 != L/dx:
		L = dx * (nx-1)
	
	x = np.linspace(0, L, nx)
	
	
	r = ru_0 * np.power((rd_0/ru_0), x/L)
	drdz = (ru_0 * np.log(rd_0/ru_0) * np.power((rd_0/ru_0), x/L))/L

	if len(dict_ano) == 0:
		pass
	else:
		start = dict_ano["start"]
		ano_length = dict_ano["size"]
		if dict_ano["aneurysm"] == True:
			bump = dict_ano["amp"] * ru_0
		else:
			bump = -dict_ano["amp"] *ru_0
		mu = start+ano_length/2
		end = start + ano_length
		r = r + HicksHenneBump(x, start, start+ano_length, start, ano_length*2, bump,  1.2e-1, False, True)#gaussian_aneurysm(x[ind_start:ind_end+1], bump, mu, ano_length/6) + np.min(np.abs(gaussian_aneurysm(x[ind_start:ind_end+1], bump, mu, ano_length/6)))
		drdz = np.zeros_like(r) + dHdx(x, start, start+ano_length, mu, ano_length, bump, False, True)  #gaussian_derivative(x[ind_start:ind_end+1], bump, mu, ano_length/6)
		
	A0 = np.pi * r**2
	
	Q = np.zeros_like(x)

	if pressure_type != 'linear' and pressure_type != 'sqrt':
		raise ValueError("The pressure model must be either 'linear' or 'sqrt'")
	
	alpha = al_artery(r, A0, pressure_type, k1, k2, k3)
	dalphadr = daldr(r, A0, pressure_type, k1, k2, k3)

	A = A0.copy()

	A_half_m = np.zeros_like(A[1:-1])
	A_half_p = np.zeros_like(A_half_m)

	Q_half_p = np.zeros_like(A_half_p)
	Q_half_m = np.zeros_like(A_half_m)

	A0_half_m = 0.5*(A0[1:-1] + A0[0:-2])
	A0_half_p = 0.5*(A0[1:-1] + A0[2:]) 

	drdz_half_m = 0.5*(drdz[1:-1] + drdz[0:-2])
	drdz_half_p = 0.5*(drdz[1:-1] + drdz[2:])

	alpha_half_m = 0.5*(alpha[1:-1] + alpha[0:-2])
	alpha_half_p = 0.5*(alpha[1:-1] + alpha[2:])

	dalpha_half_m = 0.5*(dalphadr[1:-1] + dalphadr[0:-2])
	dalpha_half_p = 0.5*(dalphadr[1:-1] + dalphadr[2:])

	return {
		"R1": R1,
		"R2": R2,
		"C": C,
		"A": A,
		"A_half_p": A_half_p,
		"A_half_m": A_half_m,
		"ano": True if len(dict_ano) > 0 else False,
        "Q": Q,
		"Q_half_p": Q_half_p,
		"Q_half_m": Q_half_m,
        "x": x,
        "L": L,
        "dx": dx,
		"nx": len(A),
        "alpha": alpha,
		"alpha_half_m": alpha_half_m,
		"alpha_half_p": alpha_half_p,
		"dalpha_half_m": dalpha_half_m,
		"dalpha_half_p": dalpha_half_p,
		"dalphadr": dalphadr,
		"p_type": pressure_type,
		"r": r,
		"drdz": drdz,
		"drdz_half_m": drdz_half_m,
		"drdz_half_p": drdz_half_p,
        "rho": rho,
		"mu": mu,
        "A0": A0,
		"A0_half_m": A0_half_m,
		"A0_half_p": A0_half_p,
		}

def update_time(dt, vessel):
	"""
	Updates all parameters value using Lax Wendroff Scheme

	- Parameters :
		- dt : float, Space step
		- vessel : dictionnary containing the vessel parameters
	
	- Returns :
		- A : Area at time n+1
		- Q : Blood flow rate at time n+1
	"""
	
	# Unpacking the vessel parameters
	A = vessel["A"]
	Q = vessel["Q"]
	dx = vessel["dx"]
	
	dtdx = dt/dx
	dtover2 = dt/2

	# Half time/space step 
	
	vessel["A_half_p"] = 0.5 * (A[2:] + A[1:-1]) - 0.5 * dtdx * (Q[2:] - Q[1:-1]) 
	vessel["A_half_m"] = 0.5 * (A[1:-1] + A[0:-2]) - 0.5 * dtdx * (Q[1:-1] - Q[0:-2]) 

	vessel["Q_half_p"] = 0.5 * (Q[2:] + Q[1:-1]) - 0.5* dtdx * (Flux(vessel, j = 2) - Flux(vessel, j = 1, k = -1)) \
		+ dtover2 * 0.5 * (Source(vessel, j = 2) + Source(vessel, j =1, k = -1))
	
	vessel["Q_half_m"] = 0.5 * (Q[1:-1] + Q[0:-2]) - 0.5* dtdx * (Flux(vessel, j = 1, k = -1) - Flux(vessel, j = 0, k = -2)) \
		+ dtover2 * 0.5 * (Source(vessel, j = 1, k = -1) + Source(vessel, j =0, k = -2))
	
	# Full step
	
	A[1:-1] = A[1:-1] - dtdx * (vessel["Q_half_p"] - vessel["Q_half_m"])
	Q[1:-1] = Q[1:-1] - dtdx * (Flux(vessel, j = 0, p = 1) - Flux(vessel, j = 0, p = 0))\
			+ dtover2 * (Source(vessel, j = 0, p = 1) + Source(vessel, j = 0, p = 0)) 

	return A,Q

def extrapolate(pos, x, y):
	"""
	Extrapolates the value of y at position pos using linear extrapolation
	Parameters :
	- pos : float representing the position where to extrapolate
	- x : array containing the x values (2 values only)
	- y : array containing the y values (2 values only)
	Returns :
	- y_pos : value of y at position pos
	"""
	
	if len(x) != 2 or len(y) != 2:
		raise ValueError("x and u must be arrays of length 2")
	elif x[0] == x[1]:
		raise ValueError("The values provided in x must be different to perform an extrapolation")
	else:
		return y[0] + (y[1] - y[0]) * (pos - x[0])/(x[1] - x[0])

def dFdx1p(dt, dx, mu, rho, x1, x2):
	"""
	Computes the derivative of the flux and source term with respect of the flow (x1) for the parent vessel
	Used for the computation of the Jacobian matrix in the Junction boundary condition

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : float representing the flow
		- x2 : float representing the area

	Returns:
		- Value of the derivative at point (Q,A) = (x1,x2)
	"""
	return -2 * (dt/dx) * (x1/x2) - (dt/2) * ((8*np.pi*mu)/(rho* x2))

def dFdx1d(dt, dx, mu, rho, x1, x2):
	"""
	Computes the derivative of the flux and source term with respect of the flow (x1) for the daughter vessel
	Used for the computation of the Jacobian matrix in the Junction boundary condition

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : float representing the flow
		- x2 : float representing the area

	Returns:
		- Value of the derivative at point (Q,A) = (x1,x2)
	"""
	return 2 * (dt/dx) * (x1/x2) - (dt/2) * ((8*np.pi*mu)/(rho* x2))
	
def dFdx2p(dt, dx, mu, rho, x1, x2, alpha_e_p, dalpha_e_p, drdz_e_p, A0_e_p, r0_e_p, pressure_type):
	"""
	Computes the derivative of the flux and source term with respect of the area (x2) for the parent vessel
	Used for the computation of the Jacobian matrix in the Junction boundary condition

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : float representing the flow
		- x2 : float representing the area
		- alpha_e_p : extrapolated value for the elasticity for the parent vessel
		- dalpha_e_p : extrapolated value for the derivative of alpha for the parent vessel
		- drdz_e_p : extrapolated value for the derivative of the radius for the parent vessel
		- A0_e_p : extrapolated value for the area at rest for the parent vessel
		- r0_e_p : extrapolated value for the radius at rest for the parent vessel
		- pressure_type : pressure model used in the simulation

	Returns:
		- Value of the derivative at point (Q,A) = (x1,x2)
	"""
	if pressure_type == 'linear':
		return -(dt/dx) * (-(x1/x2)**2 + (alpha_e_p*x2/rho)) + (dt/(2 * rho)) * ((8*np.pi*mu * x1)/(x2**2) + dbdxx(x2, alpha_e_p, r0_e_p, drdz_e_p, dalpha_e_p, A0_e_p))
	else:
		return -(dt/dx) * (-(x1/x2)**2 + (alpha_e_p*np.sqrt(A0_e_p))/(2*rho*np.sqrt(x2))) \
			+ (dt/(2 * rho)) * ((8*np.pi*mu * x1)/(x2**2) + ((1/(2*np.sqrt(x2)))*(alpha_e_p*np.sqrt(np.pi) + dalpha_e_p*np.sqrt(A0_e_p)) - dalpha_e_p)*drdz_e_p) #dfdx2p

def dFdx2d(dt, dx, mu, rho, x1, x2, alpha_e_d, dalpha_e_d, drdz_e_d, A0_e_d, r0_e_d, pressure_type):
	"""
	Computes the derivative of the flux and source term with respect of the area (x2) for the daughter vessel
	Used for the computation of the Jacobian matrix in the Junction boundary condition

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : float representing the flow
		- x2 : float representing the area
		- alpha_e_d : extrapolated value for the elasticity for the daughter vessel
		- dalpha_e_d : extrapolated value for the derivative of alpha for the daughter vessel
		- drdz_e_d : extrapolated value for the derivative of the radius for the daughter vessel
		- A0_e_d : extrapolated value for the area at rest for the daughter vessel
		- r0_e_d : extrapolated value for the radius at rest for the daughter vessel
		- pressure_type : pressure model used in the simulation

	Returns:
		- Value of the derivative at point (Q,A) = (x1,x2)
	"""
	if pressure_type == 'linear':
		return -(dt/dx) * ((x1/x2)**2 - (alpha_e_d*x2/rho)) + (dt/(2 * rho)) * ((8*np.pi*mu * x1)/(x2**2) + dbdxx(x2, alpha_e_d, r0_e_d, drdz_e_d, dalpha_e_d, A0_e_d)) #dfdx2d
	else:
		return -(dt/dx) * ((x1/x2)**2 - (alpha_e_d*np.sqrt(A0_e_d))/(2*rho*np.sqrt(x2))) \
			+ (dt/(2 * rho)) * ((8*np.pi*mu * x1)/(x2**2) + ((1/(2*np.sqrt(x2)))*(alpha_e_d*np.sqrt(np.pi) + dalpha_e_d*np.sqrt(A0_e_d)) - dalpha_e_d)*drdz_e_d) #dfdx2d

def dPdx(alpha, a0, x, pressure_type):
	"""
	Computes the derivative of the pressure with respect to the area (x)
	Used for the computation of the Jacobian matrix

	Parameters :
		- alpha : elasticity parameter
		- a0 : value of the area at rest
		- x : value of the area 
		- pressure_type : string of the pressure model used in the simulation
	
	Returns:
		- Value of the pressure derivative value at point x 
	"""
	if pressure_type == 'linear':
		return alpha
	else:
		return (alpha/2)*np.sqrt(a0/x**3)

def dbdxx(x, alpha, r, dr, dal, A0):
	"""
	Computes the value of d^2B/dzdx used for the jacobian matrix

	Parameters:
		- x : value of the area 
		- alpha : float, elasticity parameter
		- r : float, radius at rest
		- dr : float, spatial derivative of the radius at rest
		- dal : float, spatial derivative of the elasticty
		- A0 : float, cross sectionnal area at rest
	
	Returns:
		- Value of d^2B/dzdx at point x
	"""
	return 2*(alpha * np.pi * r - (x - A0)*dal)*dr

def Richt_q_p(dt, dx, mu, rho, x1, x2, x3, vessel, alpha_e_p, dalpha_e_p, drdz_e_p, A0_e_p, R0_e_p, Q_p_half, A_p_half, pressure_type):
	"""
	Richtmyer computation of the flow for the parent vessel adapted for extrapolated value, used to get the value of some residual coefficients

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : value of flow at time n and space M
		- x2 : value of flow at time n+1/2 and space M+1/2
		- x3 : value of area at time n+1/2 and space M+1/2
		- vessel : dictionnary containing some vessel parameter for space point M-1/2
		- alpha_e_p : extrapolated value for the elasticity for the parent vessel
		- dalpha_e_p : extrapolated value for the derivative of alpha for the parent vessel
		- drdz_e_p : extrapolated value for the derivative of the radius for the parent vessel
		- A0_e_p : extrapolated value for the area at rest for the parent vessel
		- R0_e_p : extrapolated value for the radius at rest for the parent vessel
		- Q_p_half : computed value of flow at time n+1/2 and space M-1/2
		- A_p_half : computed value of area at time n+1/2 and space M-1/2
		- pressure_type : pressure model used in the simulation

	Returns :
		- Value of residual obtained with richtmyer scheme for the parent vessel at point (x1,x2,x3)
	"""
	if pressure_type == 'linear':
		return - x1 + vessel["Q"][-1] \
			- (dt/dx) * (x2**2/x3 + (alpha_e_p/(2*rho))*(x3**2 - A0_e_p**2) - Q_p_half**2/A_p_half - (vessel["alpha_half_p"][-1]/(2*rho))*(A_p_half**2 - vessel["A0_half_p"][-1]**2))\
			+ (dt/(2* rho)) * (-(8*np.pi*mu) *(Q_p_half/A_p_half + x2/x3) + alpha_e_p * (x3 - A0_e_p)*(2*np.pi*R0_e_p - ((x3 - A0_e_p)/alpha_e_p)*dalpha_e_p)*drdz_e_p \
			+ vessel["alpha_half_p"][-1]*(vessel["A0_half_p"][-1] - vessel["A0_half_p"][-1])*(2*np.pi*vessel["r"][-1] - ((vessel["A_half_p"][-1] - vessel["A0_half_p"][-1])/vessel["alpha_half_p"][-1])*vessel["dalpha_half_p"][-1])*vessel["drdz_half_p"][-1])
	else:	
		return - x1 + vessel["Q"][-1] \
			- (dt/dx) * (x2**2/x3 + (alpha_e_p/rho)*(np.sqrt(x3*A0_e_p)) - Q_p_half**2/A_p_half - (vessel["alpha_half_p"][-1]/(rho))*(np.sqrt(A_p_half*vessel["A0_half_p"][-1])))\
			+ (dt/(2* rho)) * (-(8*np.pi*mu) *(Q_p_half/A_p_half + x2/x3) + (2*np.sqrt(x3)*(np.sqrt(np.pi) * alpha_e_p + np.sqrt(A0_e_p)*dalpha_e_p)-x3*dalpha_e_p)*drdz_e_p \
			+ (2*np.sqrt(A_p_half)*(np.sqrt(np.pi)*vessel["alpha_half_p"][-1] + np.sqrt(vessel["A0_half_p"][-1])*vessel["dalpha_half_p"][-1])- A_p_half*vessel["dalpha_half_p"][-1])*vessel["drdz_half_p"][-1])

def Richt_q_d(dt, dx, mu, rho, x1, x2, x3, vessel, alpha_e_d1, dalpha_e_d1, drdz_e_d1, A0_e_d1, R0_e_d1, Q_d1_half, A_d1_half, pressure_type):
	"""
	Richtmyer computation of the flow for the daughter vessel adapted for extrapolated value, used to get the value of some residual coefficients

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- mu : float, blood viscosity
		- rho : float, blood density
		- x1 : value of flow at time n and space M
		- x2 : value of flow at time n+1/2 and space M-1/2
		- x3 : value of area at time n+1/2 and space M-1/2
		- vessel : dictionnary containing some vessel parameter for space point M+1/2
		- alpha_e_d1 : extrapolated value for the elasticity for the daughter vessel
		- dalpha_e_d1 : extrapolated value for the derivative of alpha for the daughter vessel
		- drdz_e_d1 : extrapolated value for the derivative of the radius for the daughter vessel
		- A0_e_d1 : extrapolated value for the area at rest for the daughter vessel
		- R0_e_d1 : extrapolated value for the radius at rest for the daughter vessel
		- Q_d1_half : computed value of flow at time n+1/2 and space M+1/2
		- A_d1_half : computed value of area at time n+1/2 and space M+1/2
		- pressure_type : pressure model used in the simulation

	Returns :
		- Value of residual obtained with richtmyer scheme for the daughter vessel at point (x1,x2,x3)
	"""
	if pressure_type == 'linear':
		return - x1 + vessel["Q"][0] \
			- (dt/dx) * (Q_d1_half**2/A_d1_half + (vessel["alpha_half_p"][0]/(2*rho))*(A_d1_half**2 - vessel["A0_half_p"][0]**2)- x2**2/x3 - (alpha_e_d1/(2*rho))*(x3**2 - A0_e_d1**2))\
			+ (dt/(2* rho)) * (-(8*np.pi*mu) *(Q_d1_half/A_d1_half + x2/x3) + alpha_e_d1 * (x3 - A0_e_d1)*(2*np.pi*R0_e_d1 - ((x3 - A0_e_d1)/alpha_e_d1)*dalpha_e_d1)*drdz_e_d1 \
			+ vessel["alpha_half_p"][0]*(vessel["A0_half_p"][0] - vessel["A0_half_p"][0])*(2*np.pi*vessel["r"][0] - ((vessel["A_half_p"][0] - vessel["A0_half_p"][0])/vessel["alpha_half_p"][0])*vessel["dalpha_half_p"][0])*vessel["drdz_half_p"][0])

	else:
		return - x1 + vessel["Q"][0] \
			- (dt/dx) * (Q_d1_half**2/A_d1_half + (vessel["alpha_half_p"][0]/(rho))*(np.sqrt(A_d1_half*vessel["A0_half_p"][0]))- x2**2/x3 - (alpha_e_d1/(rho))*(np.sqrt(x3*A0_e_d1)))\
			+ (dt/(2* rho)) * (-(8*np.pi*mu) *(Q_d1_half/A_d1_half + x2/x3) + (2*np.sqrt(x3)*(np.sqrt(np.pi) * alpha_e_d1 + np.sqrt(A0_e_d1)*dalpha_e_d1)-x3*dalpha_e_d1)*drdz_e_d1 \
			+ (2*np.sqrt(A_d1_half)*(np.sqrt(np.pi)*vessel["alpha_half_p"][0] + np.sqrt(vessel["A0_half_p"][0])*vessel["dalpha_half_p"][0])- A_d1_half*vessel["dalpha_half_p"][0])*vessel["drdz_half_p"][0])

def Richt_a_p(dt, dx, x1, x2, A_out, Q_p_half):
	"""
	Computation of the residual using the richtmyer scheme's step 2 for area for the parent vessel

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- x1 : variable, Area at time n+1, space M of parent vessel
		- x2 : variable, Flow at time n+1/2, space M+1/2 of parent vessel
		- A_out : float, area at time n and space M of parent vessel
		- Q_p_half : float, flow at time n+1/2, space M-1/2 of parent vessel
	
	Returns:
		- Value of residual for continuity equation of parent vessel
	"""
	return -x1 + A_out - (dt/dx) * (x2 - Q_p_half)

def Richt_a_d(dt, dx, x1, x2, A_in, Q_d1_half):
	"""
	Computation of the residual using the richtmyer scheme's step 2 for area for the daughter vessel

	Parameters:
		- dt : float, time step
		- dx : float, space step
		- x1 : variable, Area at time n+1, space M of daughter vessel
		- x2 : variable, Flow at time n+1/2, space M-1/2 of daughter vessel
		- A_in : float, area at time n and space M of daughter vessel
		- Q_d1_half : float, flow at time n+1/2, space M+1/2 of daughter vessel
	
	Returns:
		- Value of residual for continuity equation of daughter vessel
	"""
	return	-x1 + A_in - (dt/dx)*(Q_d1_half - x2)

def ghost_point(x1, x2, p_half):
	"""
	Computing the residual from the ghost point method 

	Parameters:
		- x1 : variable, either A or Q at time n+1/2, space M
		- x2 : variable, either A or Q at time n+1/2, space M+1/2 for parent vessel M-1/2 for daughter vessel
		- p_half : float, either A or Q at time n+1/2, space M-1/2 for parent vessel M+1/2 for daughter vessel

	Returns:
		-Value of the residual using the ghost point method
	"""
	return -x1 + (x2 + p_half)/2

def pre_nr(alpha, x1, a0, pressure_type):
	"""
	Computation of the transmural pressure using the variables from the bifurcation boundary conditions
	It is used to compute the residuals
	
	Parameters:
		- alpha : float, elasticity parameter
		- x1 : variable, correspond to the area
		- a0 : float, area at rest
		- pressure_type : pressure model used in the current simulation

	Returns:
		- Value of the pressure at point x1
	"""
	if pressure_type == 'linear':
		return alpha*(x1 - a0) 
	else:
		return alpha*(1-np.sqrt(a0/x1))

def create_Jacobian(x, list_parent, list_daughter, dt, N, alpha_dict, dalpha_dict, drdz_dict, A0_dict, r0_dict):
	"""
	Creates the Jacobian matrix used for solving branching condition with Newton-Raphson method
	
	Parameters:
		- x : array, list of unknowns | solution vector we are trying to solve
		- list_parent : dictionnary of parent vessels concerned by the actual boundary condition
		- list_daughter : dictionnary of daughter vessels concerned by the actual boundary condition
		- dt : float, time step
		- N : int, number of unknowns
		- alpha_dict : dictionnary of extrapolated value of alpha for each vessel
		- dalpha_dict : dictionnary of extrapolated value of dalpha/dr for each vessel
		- drdz_dict : dictionnary of extrapolated value of dr/dz for each vessel
		- A0_dict : dictionnary of extrapolated value of area at rest for each vessel
		- r0_dict : dictionnary of extrapolated value of radius at rest for each vessel

	Returns:
		- JF : Jacobian matrix evaluated at point x
	"""
	JF = np.zeros((N,N))

	dx = list_parent[0]["dx"]
	mu = list_parent[0]["mu"]
	rho = list_parent[0]["rho"]
	pressure_type = list_parent[0]["p_type"]
	first_key = next(iter(list_parent))
	dtdx = dt/dx

	n_p = len(list_parent)
	n_d = len(list_daughter)
	M = n_p + n_d
	
	# Diag from richtmyer scheme + ghost point partial derivative (eqs. 0 to 4*M-1 and derivative with respect of 0th to 4*M-1 variable)
	for i in range(4*M):
		JF[i,i] = -1.0
	
	# Coefficients for flow conservation (half and full time ver.) (eqs. 4*M and 4*M+1 and 
	# derivative with respect of 0, ..., M variable for full and 2*M,..., 3*M variable for half)
	ind_row_half = 4*M
	ind_row_full = ind_row_half+1
	ind_col_half = 2*M
	ind_col_full = 0
	for i in range(n_p):
		JF[ind_row_half,ind_col_half+i] = -1
		JF[ind_row_full,ind_col_full+i] = -1
	for i in range(n_d):	
		JF[ind_row_half,ind_col_half+n_p+i] = 1
		JF[ind_row_full,ind_col_full+n_p+i] = 1

	# Coefficient for Richtmyer scheme, mass conservation (eq M to 2*M-1), derivative with respect to 4*M to 5*M-1 variable

	for i in range(0,M,1):
		if i < n_p:
			JF[M+i,4*M+i] = -dtdx
		else:
			JF[M+i,4*M+i] = dtdx
	
	# Coefficient for Ghost Point method, (eqs 2*M to 4*M-1), derivative with respect to 4*M to 6*M-1 variables
	for i in range(0,2*M,1):
		JF[2*M+i, 4*M+i] = 0.5

	# Derivative of flux with respect of flow for Jacobian (eqs 0 to M-1), derivative with respect of 4*M to 5*M-1 variable
	for i in range(0,M,1):
		if i < n_p:
			JF[i, 4*M+i] = dFdx1p(dt, dx, mu, rho, x[4*M+i], x[5*M+i])
		else:
			JF[i, 4*M+i] = dFdx1d(dt, dx, mu, rho, x[4*M+i], x[5*M+i])

	# Derivative of flux with respect of area for Jacobian (eqs 0 to M-1), derivative with respect of 5*M to 6*M-1 variable
	for i in range(0,M,1):
		if i < n_p:
			JF[i, 5*M+i] = dFdx2p(dt, dx, mu, rho, x[4*M+i], x[5*M+i], alpha_dict[i], dalpha_dict[i], drdz_dict[i], A0_dict[i], r0_dict[i], pressure_type)
		else:
			JF[i, 5*M+i] = dFdx2d(dt, dx, mu, rho, x[4*M+i], x[5*M+i], alpha_dict[i], dalpha_dict[i], drdz_dict[i], A0_dict[i], r0_dict[i], pressure_type)

	items = list(list_parent.items())
	first_dict = list_parent[first_key]
	rest_parent_dict = dict(items[1:]) if len(list_parent) > 1 else {}
	# Partial derivative for pressure conservation, half time step (eqs. 4*M+2 to 4*M+dp+1)
	i = 0
	j = n_p
	for vessel_name, vessel in chain(rest_parent_dict.items(), list_daughter.items()):
		JF[4*M+2+i, 3*M] = -dPdx(first_dict["alpha"][-1], first_dict["A0"][-1], x[3*M], pressure_type)
		if vessel_name in list_parent:
			JF[4*M+2+i, 3*M+1+i] = dPdx(vessel["alpha"][-1], vessel["A0"][-1], x[3*M+1+i], pressure_type)
		else:
			JF[4*M+2+i, 3*M+j] = dPdx(vessel["alpha"][0], vessel["A0"][0], x[3*M+j], pressure_type)
			j = j + 1
		i = i + 1


	# Partial derivative for pressure conservation, full time step (eqs. 4*M+dp+2 to 6M-1)
	i = 0
	j = n_p
	for vessel_name, vessel in chain(rest_parent_dict.items(), list_daughter.items()):
		JF[5*M+1+i, M] = -dPdx(first_dict["alpha"][-1], first_dict["A0"][-1], x[M], pressure_type)
		if vessel_name in list_parent:
			JF[5*M+1+i, M+1+i] = dPdx(vessel["alpha"][-1], vessel["A0"][-1], x[M+1+i], pressure_type)
		else:
			JF[5*M+1+i, M+j] = dPdx(vessel["alpha"][0], vessel["A0"][0], x[M+j], pressure_type)
			j = j + 1
		i = i + 1
	#for i in range(N):
	#	print("Line JF : " + str(i) + "\n", JF[i,:])
	return JF

def Func(x, list_parent, list_daughter, list_flow_parent, list_flow_daughter, list_area_parent, list_area_daughter, dt, N, alpha_dict, dalpha_dict, drdz_dict, A0_dict, r0_dict):
	"""
	Residuals for the Newton Raphson method for the bifurcation boundary condition

	Parameters : 
	- x : array containing the values of the variables at the bifurcation
	- list_parent : dictionnary of parent vessels concerned by the actual boundary condition
	- list_daughter : dictionnary of daughter vessels concerned by the actual boundary condition
	- list_flow_parent | list_flow_daughter : dictionnaries of flow values computed with the first step of the scheme for each parent | daughter vessels
	- list_area_parent | list_area_daughter : dictionnaries of area values computed with the first step of the scheme for each parent | daughter vessels
	- dt : float, time step
	- N : int, number of unknowns
	- alpha_dict : dictionnary of extrapolated value of alpha for each vessel
	- dalpha_dict : dictionnary of extrapolated value of dalpha/dr for each vessel
	- drdz_dict : dictionnary of extrapolated value of dr/dz for each vessel
	- A0_dict : dictionnary of extrapolated value of area at rest for each vessel
	- r0_dict : dictionnary of extrapolated value of radius at rest for each vessel
	
	Returns : 
		- F : array containing the residuals computed at point x
	"""
	
	F = np.zeros(N)
	n_p = len(list_parent)
	n_d = len(list_daughter)
	M = n_p + n_d
	M2 = 2*M
	M3 = 3*M
	M4 = 4*M
	M5 = 5*M
	first_key = next(iter(list_parent))

	pressure_type = list_parent[first_key]["p_type"]
	dx = list_parent[first_key]["dx"]
	mu = list_parent[first_key]["mu"]
	rho = list_parent[first_key]["rho"]

	# F0 to FM-1 : Richtmeyer Scheme for Flow conservation
	# FM to F2M-1 : Richtneyer scheme for area
	# F2M to F3M-1 : Ghost point method for flow
	# F3M to F4M-1 : Ghost point method for area

	for i in range(0,n_p,1): # loop for parent vessels
		F[i] = Richt_q_p(dt, dx, mu, rho, x[i], x[M4+i], x[M5+i], list_parent[i], alpha_dict[i], dalpha_dict[i], drdz_dict[i], A0_dict[i], r0_dict[i], list_flow_parent[i], list_area_parent[i], pressure_type)
		F[M+i] = Richt_a_p(dt, dx, x[M+i], x[M4+i], list_parent[i]["A"][-1], list_flow_parent[i])
		F[M2+i] = ghost_point(x[M2+i], x[M4+i], list_flow_parent[i])
		F[M3+i] = ghost_point(x[M3+i], x[M5+i], list_area_parent[i])
	
	for i in range(n_p, M, 1): # loop for daughter vessels
		F[i] = Richt_q_d(dt, dx, mu, rho, x[i], x[M4+i], x[M5+i], list_daughter[i], alpha_dict[i], dalpha_dict[i], drdz_dict[i], A0_dict[i], r0_dict[i], list_flow_daughter[i], list_area_daughter[i], pressure_type)
		F[M+i] = Richt_a_d(dt, dx, x[M+i], x[M4+i], list_daughter[i]["A"][0], list_flow_daughter[i])
		F[M2+i] = ghost_point(x[M2+i], x[M4+i], list_flow_daughter[i])
		F[M3+i] = ghost_point(x[M3+i], x[M5+i], list_area_daughter[i])

	# F4M and F4M+1 : Flow conservation at the bifurcation at half and full time step
	
	for i in range(0,n_p, 1):
		F[M4] -= x[M2+i]
		F[M4+1] -= x[i]
	for i in range(n_p, M, 1):
		F[M4] += x[M2+i]
		F[M4+1] += x[i]

	items = list(list_parent.items())
	first_dict = list_parent[first_key]
	rest_parent_dict = dict(items[1:]) if len(list_parent) > 1 else {}
	
	# F4M+2 to F5M : Pressure conservation at the bifurcation at half time step
	i = 0
	j = n_p
	for vessel_name, vessel in chain(rest_parent_dict.items(), list_daughter.items()):
		if vessel_name in rest_parent_dict:
			F[M4+2+i] = pre_nr(vessel["alpha"][-1], x[M3+1+i], vessel["A0"][-1], pressure_type) - pre_nr(first_dict["alpha"][-1], x[M3], first_dict["A0"][-1], pressure_type)
		else:
			F[M4+2+i] = pre_nr(vessel["alpha"][0], x[M3+j], vessel["A0"][0], pressure_type) - pre_nr(first_dict["alpha"][-1], x[M3], first_dict["A0"][-1], pressure_type)
			j = j + 1
		i = i + 1


	# F5M+1 to F6M-1 : Pressure conservation at the bifurcation at full time step
	i = 0
	j = n_p
	for vessel_name, vessel in chain(rest_parent_dict.items(), list_daughter.items()):
		if vessel_name in rest_parent_dict:
			F[M5+1+i] = pre_nr(vessel["alpha"][-1], x[M+1+i], vessel["A0"][-1], pressure_type) - pre_nr(first_dict["alpha"][-1], x[M], first_dict["A0"][-1], pressure_type)
		else:
			F[M5+1+i] = pre_nr(vessel["alpha"][0], x[M+j], vessel["A0"][0], pressure_type) - pre_nr(first_dict["alpha"][-1], x[M], first_dict["A0"][-1], pressure_type)
			j = j + 1
		i = i + 1
	#for i in range(np.shape(F)[0]):
	#	print("Line F : " + str(i) + "\n", F[i])
	#raise NotImplementedError
	return F

def junction_1_2(list_parent, list_daughter, dt):
	"""
	Computes the junction boundary condition between n_p parent vessels and n_d daughter vessels.
	
	Parameters :
		- list_parent : dictionnary of parent vessels concerned by the actual boundary condition
		- list_daughter : dictionnary of daughter vessels concerned by the actual boundary condition
		- dt : time step

	Returns :
		- main_vessel, d1_vessel, d2_vessel : updated dictionaries with new values for A and Q
	"""
	
	num_p = len(list_parent)
	num_d = len(list_daughter)

	M = num_p + num_d
	N = 6*M
	x = np.zeros(N)

	list_area_parent = {}
	list_flow_parent = {}

	for vessel_name, vessel in list_parent.items(): 
		list_area_parent[vessel_name] = 0.5 * (vessel["A"][-1] + vessel["A"][-2]) - 0.5 * (dt/vessel["dx"]) * (vessel["Q"][-1] - vessel["Q"][-2])
		list_flow_parent[vessel_name] = (0.5*(vessel["Q"][-1] + vessel["Q"][-2]) - 0.5 * (dt/vessel["dx"]) * (Flux(vessel, j = -1) - Flux(vessel,j = -2, k = -1))\
			  + (dt/4) * (Source(vessel, j = -1) + Source(vessel, j = -2, k = -1)))[0]

	list_area_daughter = {}
	list_flow_daughter = {}

	for vessel_name, vessel in list_daughter.items(): 
		list_area_daughter[vessel_name] = 0.5 * (vessel["A"][1] + vessel["A"][0]) - 0.5 * (dt/vessel["dx"]) * (vessel["Q"][1] - vessel["Q"][0])
		list_flow_daughter[vessel_name] = (0.5*(vessel["Q"][1] + vessel["Q"][0]) - 0.5 * (dt/vessel["dx"]) * (Flux(vessel, j = 1, k = 2) - Flux(vessel,j = 0, k = 1))\
			  + (dt/4) * (Source(vessel, j = 1, k = 2) + Source(vessel, j = 0, k = 1)))[0]


	# Create extrapolated value for both jacobian and residuals

	
	alpha_dict = {}
	dalpha_dict = {}
	drdz_dict = {}
	A0_dict = {}
	r0_dict = {}
	
	i = 0
	
	for key, vessel in chain(list_parent.items(), list_daughter.items()):
		if i < num_p:
			alpha_dict[i] = extrapolate(vessel["L"]+ (vessel["dx"]/2),vessel["x"][-2:],vessel["alpha"][-2:])
			dalpha_dict[i] = extrapolate(vessel["L"]+ (vessel["dx"]/2), vessel["x"][-2:],vessel["dalphadr"][-2:])
			drdz_dict[i] =  extrapolate(vessel["L"]+ (vessel["dx"]/2), vessel["x"][-2:],vessel["drdz"][-2:])
			A0_dict[i] = extrapolate(vessel["L"] + (vessel["dx"]/2), vessel["x"][-2:], vessel["A0"][-2:])
			r0_dict[i] = np.sqrt(A0_dict[i]/np.pi)
		else:
			alpha_dict[i] = extrapolate(-(vessel["dx"]/2), vessel["x"][:2],vessel["alpha"][:2])
			dalpha_dict[i] = extrapolate(-(vessel["dx"]/2), vessel["x"][:2],vessel["dalphadr"][:2])
			drdz_dict[i] =  extrapolate(-(vessel["dx"]/2), vessel["x"][:2],vessel["drdz"][:2])
			A0_dict[i] = extrapolate(-(vessel["dx"]/2), vessel["x"][:2], vessel["A0"][:2])
			r0_dict[i] = np.sqrt(A0_dict[i]/np.pi)

		i += 1
	 
	"Setting up the initial guess for Newton Raphson method"
	
	i = 0
	for vessel_name, vessel in list_parent.items(): 
		x[i] = list_flow_parent[vessel_name]
		x[i+M] = list_area_parent[vessel_name]
		x[i+2*M] = (vessel["Q"][-1] + vessel["Q"][-2])/2
		x[i+3*M] = (vessel["A"][-1] + vessel["A"][-2])/2
		x[i+4*M] = vessel["Q"][-1]
		x[i+5*M] = vessel["A"][-1]
		i+=1

	i=0
	for vessel_name, vessel in list_daughter.items(): 
		x[i+num_p] = list_flow_daughter[vessel_name]
		x[i+M+num_p] = list_area_daughter[vessel_name]
		x[i+2*M+num_p] = (vessel["Q"][-1] + vessel["Q"][-2])/2
		x[i+3*M+num_p] = (vessel["A"][-1] + vessel["A"][-2])/2
		x[i+4*M+num_p] = vessel["Q"][-1]
		x[i+5*M+num_p] = vessel["A"][-1]
		i+=1
	
	"""Iterations part"""
	k = 0
	while k < 1000:
		JF = create_Jacobian(x, list_parent, list_daughter,dt, N, alpha_dict, dalpha_dict, drdz_dict, A0_dict, r0_dict)
		JF_inv = np.linalg.inv(JF)
		f = Func(x, list_parent, list_daughter, list_flow_parent, list_flow_daughter, list_area_parent, list_area_daughter, dt, N, alpha_dict, dalpha_dict, drdz_dict, A0_dict, r0_dict)
		x1 = x - np.dot(JF_inv, f)
		test = np.linalg.norm(x1 -x)
		if  test < 1e-8:
			break
		k += 1
		np.copyto(x, x1)
	y = x[:2*M]
	return y


if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		raise ValueError("A config file must be mentionned")
	else:
		ves, sim, param, connectivity_tab = cr.read_config(sys.argv[1])
	
	# Blood properties

	dx = sim["dx"]							# Space step
	rho = param["rho"]						# Blood density [g/cm^3]
	mu = param["mu"]						# Blood viscosity [g/(cm.s)] 
	diastolic_pressure = param["diast"]		# Average diastolic pressure [mmHg]

	""" 
	Choose the pressure model you want to use for the simulation
	- Linear : p(A) = Eh/(2 * A0) * (A-A0)
	- Sqrt : p(A) = (Eh/r0) * (1- sqrt(A0/A))
	"""
	pressure_model = ves["pressure_model"]		# Linear or sqrt

    # 3WK BC parameters

	R1 = param["r1"]				# First resistance [g/s * cm^{-4}]
	R2 = param["r2"]				# Second resistance [g/s * cm^{-4}]
	C = param["c"]					# Capacitance coefficient [cm^4.s^2/g]

	# Elasticity parameter 

	k1 = param["k1"]				# [g/(s² cm)]
	k2 = param["k2"]				# [cm^-1]
	k3 = param["k3"]				# [g/(s² cm)]

	# Declare number of vessels in total (including the parent vessel)
	
	n_vessels = ves["number_vessels"] # Number of vessels 1 (no bifurcation, only parent vessel) or 3 (junction 1 to 2)

	# Setting anomalies in the vessel

	stenosis = False 				# Set to True if there is a stenosis in the vessel
	aneurysm = False 				# Set to True if there is an aneurysm in the vessel

	if ves["anomalies"] == 'Stenosis':
		stenosis = True
	elif ves["anomalies"] == 'Aneurysm':
		aneurysm = True
	elif ves["anomalies"] != 'None':
		raise ValueError("The anomalies parameter in the config file must be one of the following : \n 'None' , 'Stenosis' , 'Aneurysm' ")
	
	if aneurysm or stenosis:
		size_anomalies = ves["size_anomaly"]		# size of anomalies in cm
		start_ano = ves["start"]					# beginning of ano
		bump_amp = ves["bump"]						# percentage of the bump
		dict_ano = {'size':size_anomalies, 'start':start_ano, 'amp':bump_amp, 'stenosis':stenosis, 'aneurysm':aneurysm}
	else:
		dict_ano = {}

	dict_vessels = {} 

	for vessel in range(int(n_vessels)):
		str_ves = "vessel" + str(vessel)
		if vessel == 0:
			if n_vessels == 1:
				dict_vessels[str_ves] = create_vessel(R1, R2, C, ves["length"], ves["ru"], ves["rd"], dx, mu, rho, pressure_model, dict_ano, k1, k2, k3)
			else:
				dict_vessels[str_ves] = create_vessel(R1, R2, C, ves["length"][0], ves["ru"][0], ves["rd"][0], dx, mu, rho, pressure_model, dict_ano, k1, k2, k3)
		else:
			dict_vessels[str_ves] = create_vessel(R1, R2, C, ves["length"][vessel], ves["ru"][vessel], ves["rd"][vessel], dx, mu, rho, pressure_model, {}, k1, k2, k3)
	
	nodes = len(connectivity_tab)

	# Time discretisation

	dt = sim["dt"]						# Time step [cm] (should be using CFL condition)
	n_cycle = sim["cycles"]				# Number of cardiac cycle | At least 4 cycles for convergence in the plot
	T = sim["t"]
	heart_rate = 60/T					# Number of bpm 
	T_final = n_cycle*T					# Time of simulation [s]
	nt = round(T_final/dt)				# Number of time steps
	t = np.linspace(0,T_final, nt)  	# Time discretisation 
	
	# Setting Boundary condition 

	inlet = sim['inlet'] 			# type : str to choose between pressure and flow
	outlet = sim['outlet']			# type : str to choose between pressure and 3wk

	data_inlet = get_data(sim["path"], inlet, t, hr = heart_rate) # data_inlet : (function) Inlet flow rate [cm^3/s] or tranmural pressure [Pa] depending on choosen inlet bc
	
	# Plot parameters and variable used for visualisation

	plot_graphs = False				# Set to True if you want to see the graphs after the simulation
	time_interval = 500 			# Number of time steps between each plot update 
	t_min_plot = 3*T 				# Minimum time to plot [s] (Start of second cycle)
	t_max_plot = 4*T 				# Maximum time to plot [s] (End of second cycle)
	index_min = int(t_min_plot/dt) 	# Index corresponding to t_min_plot
	index_max = int(t_max_plot/dt) 	# Index corresponding to t_max_plot
	index = 0 						# Index used to plot the graphs

	time_plot = np.linspace(t_min_plot, t_max_plot, ceil((index_max-index_min)/time_interval)) # Time array used for plotting
	
	if inlet == 'pressure' and (sim["path"] == 'None' or sim["path"] == ''):
		data_inlet = data_inlet(t)*1333.22
	if outlet == 'pressure' and (sim['path'] == 'None' or sim["path"] == ''):
		data_outlet = data_inlet/2
	if outlet == '3wk':
		data_outlet = np.zeros_like(t)
	
	# Defining inlet and outlet function depending on the chosen boundary conditions 
	if inlet == 'flow':
		inlet_fct = inlet_bc
	elif inlet == 'pressure':
		inlet_fct = pressure_inlet
	else:
		raise ValueError("The inlet boundary condition has to be either 'pressure' or 'flow' while it is set as : " + str(inlet))

	if outlet == '3wk':
		outlet_fct = WK_outlet_bc
	elif outlet == 'pressure':
		outlet_fct = pressure_outlet
	else:
		raise ValueError("The outlet boundary condition has to be either 'pressure' or '3wk' while it is set as : " + str(outlet))
	
	dict_data = {}
	dict_inlet = {}
	dict_outlet = {}

	for vessel_name, vessel in dict_vessels.items():
		dict_data["A_"+vessel_name+"_data"] = np.zeros((len(time_plot), dict_vessels[vessel_name]["nx"]))
		dict_data["P_"+vessel_name+"_data"] = np.zeros((len(time_plot), dict_vessels[vessel_name]["nx"]))
		dict_data["Q_"+vessel_name+"_data"] = np.zeros((len(time_plot), dict_vessels[vessel_name]["nx"]))

		dict_inlet["A_"+vessel_name] = vessel["A0"][0]
		dict_inlet["Q_"+vessel_name] = 0

		dict_outlet["A_"+vessel_name] = vessel["A0"][-1]
		dict_outlet["Q_"+vessel_name] = 0

	plt.figure(figsize=(10, 6))
	for n in tqdm(range(1,nt)): # tqdm is here to give an approximation of the time left, so we dont wait in front of the screen doing nothing :)
		for node in range(nodes):
			if not connectivity_tab[node]["parents"]:
				vessel_name = "vessel" + str(connectivity_tab[node]["daughters"][0])
				dict_inlet["A_"+vessel_name], dict_inlet["Q_"+vessel_name] = inlet_fct(dict_vessels[vessel_name], [(data_inlet[n]+data_inlet[n-1])/2, data_inlet[n]], dt)
			elif not connectivity_tab[node]["daughters"]:
				vessel_name = "vessel" + str(connectivity_tab[node]["parents"][0])
				dict_outlet["A_"+vessel_name], dict_outlet["Q_"+vessel_name] = outlet_fct(vessel = dict_vessels[vessel_name], dt = dt, diast = diastolic_pressure, data_outlet = [(data_outlet[n]+data_outlet[n-1])/2, data_outlet[n]])
			else:
				dict_parents = {}
				dict_daughters = {}
				i = 0
				for key in connectivity_tab[node]["parents"]:
					dict_parents[i] = dict_vessels["vessel"+str(key)]
					i +=1
				for key in connectivity_tab[node]["daughters"]:
					dict_daughters[i] = dict_vessels["vessel"+str(key)]
					i +=1
				n_p = len(dict_parents)
				M = n_p + len(dict_daughters)
				x = junction_1_2(dict_parents, dict_daughters, dt)
				j = 0
				for key in connectivity_tab[node]["parents"]:
					dict_outlet["Q_vessel"+str(key)] = x[j]
					dict_outlet["A_vessel"+str(key)] = x[M+j]
					j += 1
				j = 0
				for key in connectivity_tab[node]["daughters"]:
					dict_inlet["Q_vessel"+str(key)] = x[n_p+j]
					dict_inlet["A_vessel"+str(key)] = x[M+n_p+j]
					j +=1

		for vessel_name, vessel in dict_vessels.items():

			vessel["A"], vessel["Q"] = update_time(dt, vessel)
			#Updating inlet and outlet
			vessel["A"][0] = dict_inlet["A_"+vessel_name]
			vessel["Q"][0] = dict_inlet["Q_"+vessel_name]

			vessel["A"][-1] = dict_outlet["A_"+vessel_name]
			vessel["Q"][-1] = dict_outlet["Q_"+vessel_name]

		if n % time_interval == 0:
			# Live plot (optional), it tends to slow down the simulation + window might superpose with other windows

			plt.clf()

			plt.subplot(4, 1, 1)
			plt.plot(dict_vessels["vessel0"]["x"], dict_vessels["vessel0"]["A"], 'b-', linewidth=1.5)
			plt.title(f'Cross-sectional Area (A) at t = {n * dt:.2f} s')
			plt.xlabel('Position (cm)')
			plt.ylabel('A (cm²)')
			plt.xlim([0, dict_vessels["vessel0"]["L"]])
			plt.ylim([0.3 , 0.5])
			plt.grid(True)

			plt.subplot(4, 1, 2)
			plt.plot(dict_vessels["vessel0"]["x"], dict_vessels["vessel0"]["Q"], 'r-', linewidth=1.5)
			plt.title(f'Flow Rate (Q) at t = {n * dt:.2f} s')
			plt.xlabel('Position (cm)')
			plt.ylabel('Q (cm³/s)')
			plt.xlim([0, dict_vessels["vessel0"]["L"]])
			if len(sys.argv) == 2:
				plt.ylim([0, 25])
			else:
				plt.ylim([0, 5])
			plt.grid(True)

			plt.subplot(4, 1, 3)
			plt.plot(dict_vessels["vessel0"]["x"], dict_vessels["vessel0"]["Q"]/dict_vessels["vessel0"]["A"], 'b-', linewidth=1.5)
			plt.title(f'Velocity (U) at t = {n * dt:.2f} s')
			plt.xlabel('Position (cm)')
			plt.ylabel('U (cm/s)')
			plt.xlim([0, dict_vessels["vessel0"]["L"]])
			plt.ylim([0, 200])
			plt.grid(True)

			plt.subplot(4, 1, 4)
			plt.plot(dict_vessels["vessel0"]["x"], pressure(dict_vessels["vessel0"]["A"], dict_vessels["vessel0"]["A0"], dict_vessels["vessel0"]["alpha"], diastolic_pressure, pressure_model) + diastolic_pressure, 'r-', linewidth=1.5)
			plt.title(f'Blood Pressure (P) at t = {n * dt:.2f} s')
			plt.xlabel('Position (cm)')
			plt.ylabel('P (mmHg)')
			plt.xlim([0, dict_vessels["vessel0"]["L"]])
			plt.ylim([20, 200])
			plt.grid(True)

			plt.pause(0.001)
		if n % time_interval == 0 and n >= index_min and n <= index_max:
			# Filling the data arrays for plotting
			for vessel_name, vessel in dict_vessels.items():
				dict_data["A_"+vessel_name+"_data"][index] = vessel["A"]
				dict_data["P_"+vessel_name+"_data"][index] = pressure(vessel["A"], vessel["A0"], vessel["alpha"], diastolic_pressure, pressure_model)
				dict_data["Q_"+vessel_name+"_data"][index] = vessel["Q"]
			index += 1
	x_plot_data = {}
	for vessel_name, vessel in dict_vessels.items():
		x_plot_data["x_"+vessel_name+"_plot"] = vessel["x"]
	
	write_data(dict_data, x_plot_data, time_plot)

	"""
	# Saving your pc memory before plotting graphs :)
	for var in list(globals()):
		if var in ('n_vessels', 'plot_graphs', '__name__', '__doc__', '__package__', '__loader__',
                   '__spec__', '__annotations__', '__builtins__'):
			continue
		if isinstance(globals()[var], type(os)):  
			continue
		del globals()[var]

	# Calling function to save graph
	t, var_dict, path = pr.get_data(n_vessels)
	pr.save_plots_3D(t, var_dict, path, plot_graphs)
	pr.save_plots_1D(t, var_dict, path, plot_graphs)
	"""