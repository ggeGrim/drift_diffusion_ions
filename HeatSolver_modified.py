#! /anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import math
from scipy.optimize import curve_fit
from scipy.integrate import ode
from matplotlib.animation import FuncAnimation
import matplotlib
import sys
import os
from pathlib import Path

class DiffusionSystem(object):
	"""
	class containing all the variables and functions to initialize, fit the data and analyse the results
	"""
	def __init__(self,):
		"""
		initializes the class variables used through the program
		"""
		self.L = 0
		self.dL = 0
		self.nL = 0
		self.t_max = 0
		self.nT = 0
		self.dT = 0
		self.De = 0
		self.extracted_e_mobility = 0
		self.elementary_charge = 1.60217662e-19	# units: C
		self.propagation_method = "exp"
		self.k_boltzmann = 1.38064852e-23	# units: J/K

	def initialize_spatial_grid(self,L, nL):
		"""
		sets the total lenght of the film and the number of grid points

		:param L: thickness of the film
		:type L: float

		:param nL: number of grid points for the evolution model
		:type nL: int

		:returns: None 
		"""
		self.L = L
		self.nL = nL
		self.dL = float(self.L)/float(self.nL)

	def initialize_time_grid(self):
		"""
		sets the total time of the simulation and the number of time steps

		:param t_max: upper time of the simulation
		:type t_max: float

		:param nT: number of time points in the simulation
		:type nT: int

		:returns: None
		"""
		self.nT = int(float(self.t_max)/float(self.dT))
		self.t = np.linspace(0, float(self.t_max), self.nT)

	def initialize_electric_field(self, electric_field):
		"""
		initializes the electric field in the film, in V/m

		:param electric_field: electron field strength in the film, in V/m
		:type electric_field: float

		:returns: None
		"""
		self.electric_field = electric_field

	def initialize_electric_field_constant(self, electric_field):
		"""
		initializes the electric field in the film, giving the same value to each point in the film. 
		The units of the values in electric_field are in V/m

		:param electric_field: electron field strength in the film, in V/m
		:type electric_field: float

		:returns: None
		"""
		self.E_vector = np.array([electric_field for i in range(self.nL)])

	def initialize_electric_field_steps(self, steps_vector):
		"""
		Initializes the electric field in the film, dividing the film in regions with different field strength. 
		The electric field values are specified by the values in steps_vector, and the number of elements of steps_vector specifies the number of regions.
		The units of the values in steps_vector are in V/m

		:param steps_vector: array of electric field strength in the film, in V/m
		:type steps_vector: array

		:returns: None
		"""

		position_vector = [i/self.nL*len(steps_vector) for i in range(self.nL)]
		region_coordinate = np.digitize(position_vector, range(len(steps_vector)))
		self.E_vector = np.array([steps_vector[i-1] for i in region_coordinate])

	def initialize_diffusion_coefficient(self, De):
		"""
		sets the diffusion coefficient for carriers in the z-direction of the film

		:param De: diffusion coefficient for carriers inside the film, in m^2/s
		:type De: float

		:returns: None
		"""
		self.De = De

	def initialize_mobility_film(self, mobility):
		"""
		initializes the mobility of electrons in the film, in cm^2/(V*s), converted in m^2/(V*s)

		:param mobility: electron mobility in the film, in cm^2/(V*s)
		:type mobility: float

		:returns: None
		"""
		self.film_e_mobility = mobility*1e-4

	def initialize_mobility_film_steps(self, steps_vector):
		"""
		Initializes the mobility of electrons in the film, dividing the film in regions with different mobility. 
		The mobility values are specified by the values in steps_vector, and the number of elements of steps_vector specifies the number of regions.
		The units of the values in steps_vector are in cm^2/(V*s), while the unit of mu_vector are converted in m^2/(V*s)

		:param steps_vector: array of electron mobility in the film, in cm^2/(V*s)
		:type steps_vector: array

		:returns: None
		"""

		position_vector = [i/self.nL*len(steps_vector) for i in range(self.nL)]
		region_coordinate = np.digitize(position_vector, range(len(steps_vector)))
		self.mu_vector = np.array([steps_vector[i-1]*1e-4 for i in region_coordinate])

	def initialize_mobility_film_constant(self, mobility):
		"""
		initializes the mobility of electrons in the film, giving the same value to each point in the film. 
		The units of the values in mobility are in cm^2/(V*s), while the unit of mu_vector are converted in m^2/(V*s)

		:param mobility: electron mobility in the film, in cm^2/(V*s)
		:type mobility: float

		:returns: None
		"""
		self.mu_vector = np.array([mobility*1e-4 for i in range(self.nL)])

	def initialize_relative_electric_charge(self, relative_charge):
		"""
		Initializes the value of the relative electric charge of the diffusing particle.
		The function initializes the attribute relative_electric_charge of the DiffusioSystem class.


		:returns: None
		"""

		self.relative_electric_charge = relative_charge

	def initialize_maxT(self,t_max):
		"""
		Initializes the value of the last time point of the simulation.
		The function initializes the attribute t_max of the DiffusioSystem class.


		:returns: None
		"""
		self.t_max = t_max

	def initialize_dT(self, dT):
		"""
		Initializes the value of the time step of the simulation.
		The function initializes the attribute dT of the DiffusioSystem class.


		:returns: None
		"""
		self.dT = dT

	def initialize_temperature(self, temperature):
		"""
		Initializes the value of the temperature of the film.
		The function initializes the attribute temperature of the DiffusioSystem class.


		:returns: None
		"""
		self.temperature = temperature

	def calculate_diffusion_coefficient(self):
		"""
		calculates the diffusion coefficient from the mobility of the film, using Einstein-Shmolkowski relationship
		The function initializes the attribute De of the DiffusioSystem class.


		:returns: None
		"""
		thermal_constant = self.k_boltzmann*self.temperature
		self.De = thermal_constant*self.film_e_mobility/(self.relative_electric_charge*self.elementary_charge)	# in m^2/s

	def calculate_diffusion_coefficient_vector(self):
		"""
		calculates the diffusion coefficient from the mobility of the film as a function of the position, using Einstein-Shmolkowski relationship
		The function initializes the attribute De_vector of the DiffusioSystem class.

		:returns: None
		"""
		thermal_constant = self.k_boltzmann*self.temperature
		self.De_vector = thermal_constant*self.mu_vector/(self.relative_electric_charge*self.elementary_charge)	# in m^2/s



	def initialize_propagation_method(self, method):
		"""
		sets the method used to propagate the evolution of the system from one time point to the next
        available methods: exponentiation method ("exp") and integrating the system of ODE with scipy.integrate solver ("integrate")

		:param method: method used to propagate the set of differential equations describing the system
		:type method: str

		:returns: None
		"""
		self.propagation_method = method

	def initialize_mode(self, mode):
		"""
		defines whether the program solves a diffusion equation ("diffusion"), a drift-diffusion equation 
        with constant coefficients ("drift-diffusion"), or a drift-diffusion equatin with varying
        coefficients ("drift_diffusion_varying_cs")

		:param mode: operational mode of the program
		:type mode: str

		:returns: None
		"""
		self.system_mode = mode

	def set_output(self, o_path):
		"""
		select the path of the output folder, if the folder doesn't exists it creates it

		:param o_path: path of the output folder
		:type o_path: string

		:returns: None
		"""
		if not os.path.isdir(o_path):
			os.makedirs(o_path)
		self.output_path = o_path


	def create_matrix_diffusion(self):
		"""
		defines the matrix used for the propagation of the system of coupled differential equations describing
		a 1D ion distribution diffusing in a grain

		:returns: None
		"""
		self.M = np.zeros((self.nL, self.nL))
		diffusion_matrix = np.zeros((self.nL,self.nL))
		diffusion_matrix += np.diag([-2*self.De/(self.dL)**2 for i in range(self.nL)])
		diffusion_matrix += np.diag([self.De/(self.dL)**2 for i in range(self.nL-1)], k=1)
		diffusion_matrix += np.diag([self.De/(self.dL)**2 for i in range(self.nL-1)], k=-1)
		diffusion_matrix[-1,-1] += self.De/(self.dL)**2
		diffusion_matrix[0,0] += self.De/(self.dL)**2
		self.M += diffusion_matrix

	def create_matrix_drift_diffusion(self):
		"""
		defines the matrix used for the propagation of the system of coupled differential equations describing
		a 1D ion distribution diffusing in a grain under an electric field

		:returns: None
		"""

		ko = (self.relative_electric_charge * self.elementary_charge * self.De * self.electric_field)/(self.k_boltzmann * self.temperature)/(2*self.dL)

		M1 = np.zeros((self.nL, self.nL))
		M2 = np.zeros((self.nL, self.nL))

		M1 += np.diag([-2*self.De/(self.dL)**2 for i in range(self.nL)])
		M1[0,0] = -self.De/(self.dL)**2	# lowest index real node
		M1[-1,-1] = -self.De/(self.dL)**2 # highest index real node
		M1 += np.diag([self.De/(self.dL)**2 for i in range(self.nL-1)], k=1)
		M1 += np.diag([self.De/(self.dL)**2 for i in range(self.nL-1)], k=-1)

		M2 += np.diag([ko for i in range(self.nL-1)], k=1)
		M2 += np.diag([-ko for i in range(self.nL-1)], k=-1)
		M2[0,0] = ko	# lowest index real node
		M2[-1,-1] = -ko	# highest index real node

		self.M = M1 + M2

	def create_matrix_drift_diffusion_varying_cs(self):
		"""
		defines the matrix used for the propagation of the system of coupled differential equations describing
		a 1D ion distribution diffusing in a grain under an electric field, allowing for spatially varying
        diffusion coefficient and electric field

		:returns: None
		"""
		# initializing the matrix of coefficients
		self.M = np.zeros((self.nL, self.nL))
		M1 = np.zeros((self.nL, self.nL))
		M2 = np.zeros((self.nL, self.nL))
		M3 = np.zeros((self.nL, self.nL))
		M4 = np.zeros((self.nL, self.nL))
		M5 = np.zeros((self.nL, self.nL))

		# matrix 1 (dD/dx * dY/dx)
		M1 = np.diag([(self.De_vector[i]-self.De_vector[i-2])/(4*self.dL**2) if i != 1 else (self.De_vector[1]-self.De_vector[0])/(4*self.dL**2) for i in range(1,self.nL)], k=1)
		M1 += np.diag([-(self.De_vector[i+1]-self.De_vector[i-1])/(4*self.dL**2) if i != self.nL-1 else -(self.De_vector[self.nL-1]-self.De_vector[self.nL-2])/(4*self.dL**2) for i in range(1,self.nL)], k=-1)
		M1[0,0] = -(self.De_vector[1]-self.De_vector[0])/(4*self.dL**2)
		M1[-1,-1] = (self.De_vector[self.nL-1]-self.De_vector[self.nL-2])/(4*self.dL**2)
		# matrix 2 (D * d2Y/dx2)
		M2 = np.diag([-2*self.De_vector[i-1]/(self.dL**2) for i in range(1,self.nL+1)])
		M2 += np.diag([self.De_vector[i-1]/(self.dL**2) for i in range(1,self.nL)], k=1)
		M2 += np.diag([self.De_vector[i]/(self.dL**2) for i in range(1,self.nL)], k=-1)
		M2[0,0] = -self.De_vector[0]/(self.dL**2)
		M2[-1,-1] = -self.De_vector[self.nL-1]/(self.dL**2)
		# matrix 3 (du/dx * E * Y)
		M3 = np.diag([(self.mu_vector[i] - self.mu_vector[i-2])/(2*self.dL)*self.E_vector[i-1] if i not in [1,self.nL] else 0 for i in range(1,self.nL+1)])
		M3[0,0] = (self.mu_vector[1] - self.mu_vector[0])/(2*self.dL)*self.E_vector[0]
		M3[-1, -1] = (self.mu_vector[self.nL-1] - self.mu_vector[self.nL-2])/(2*self.dL)*self.E_vector[self.nL-1]
		# matrix 4 (u * dE/dx * Y)
		M4 = np.diag([(self.E_vector[i] - self.E_vector[i-2])/(2*self.dL)*self.mu_vector[i-1] if i not in [1,self.nL] else 0 for i in range(1,self.nL+1)])
		M4[0,0] = (self.E_vector[1] - self.E_vector[0])/(2*self.dL)*self.mu_vector[0]
		M4[-1, -1] = (self.E_vector[self.nL-1] - self.E_vector[self.nL-2])/(2*self.dL)*self.mu_vector[self.nL-1]
		# matrix 5 (u * E* dY/dx)
		M5 = np.diag([self.mu_vector[i-1]*self.E_vector[i-1]/(2*self.dL) for i in range(1,self.nL)], k=1)
		M5 += np.diag([-self.mu_vector[i-1]*self.E_vector[i-1]/(2*self.dL) for i in range(1,self.nL)], k=-1)
		M5[0,0] = self.mu_vector[0]*self.E_vector[0]/(2*self.dL)
		M5[-1,-1] = -self.mu_vector[self.nL-1]*self.E_vector[self.nL-1]/(2*self.dL)
		# total matrix
		self.M = M1 + M2 + M3 + M4 + M5

		# # debugging lines
		# print(M4)
		# quit()
		# # end debugging lines

	def set_initial_population(self):
		"""
		set the distribution of carriers at t=0

		:returns: None
		"""
		if self.L == 0:
			print("Error encountered in creation of initial population: film length unset")
			exit()
		self.initial_population = np.zeros(self.nL)
		for i in range(self.nL):
			self.initial_population[i] = np.exp(-(i-self.nL*2/3)**2/(2*(self.nL/7)**2))	# NOTE: random example of a non-uniform distribution
		pop_sum = np.sum(self.initial_population)
		# pop_sum = 1
		self.initial_population /= pop_sum


	def propagate_ode_exp(self):
		"""
		propagates the system of coupled differential equations with the exponentiation method

		:returns: None
		"""

		# dictionary of possible systems to propagate

		if self.system_mode == 'diffusion':
			create_matrix = self.create_matrix_diffusion
		elif self.system_mode == 'drift_diffusion':
			create_matrix = self.create_matrix_drift_diffusion
		elif self.system_mode == 'drift_diffusion_varying_cs':
			create_matrix = self.create_matrix_drift_diffusion_varying_cs
		else:
			print("Program mode not recognized, quitting")
			exit()

		create_matrix()

		self.set_initial_population()

		U = spla.expm(self.M*self.dT)
		self.pop = np.zeros((len(self.initial_population),self.nT))
		self.pop[:,0] = self.initial_population

		for iT in range(1,self.nT):
			self.pop[:,iT] = np.dot(U,self.pop[:,iT-1])

		
	def propagate_ode_integrate(self):
		"""
		propagates the system of coupled differential equations integrating the ODE with scipy.integrate.ode

		:returns: None
		"""

		# dictionary of possible systems to propagate

		if self.system_mode == 'diffusion':
			create_matrix = self.create_matrix_diffusion
		elif self.system_mode == 'drift_diffusion':
			create_matrix = self.create_matrix_drift_diffusion
		elif self.system_mode == 'drift_diffusion_varying_cs':
			create_matrix = self.create_matrix_drift_diffusion_varying_cs
		else:
			print("Program mode not recognized, quitting")
			exit()

		create_matrix()

		self.set_initial_population()

		
		def fprime(t,y,M):
			return np.dot(self.M,y)

		solver = ode(fprime)
		solver.set_integrator('dopri5')
		solver.set_initial_value(self.initial_population,self.t[0])
		solver.set_f_params(self.M)

		self.pop = np.zeros((len(self.initial_population),self.nT))
		self.pop[:,0] = self.initial_population

		iT=1
		while solver.successful() and iT < self.nT:
			self.pop[:,iT] = solver.integrate(solver.t+self.dT)
			iT += 1


	def solve_DiffusionSystem(self):
		"""
		solves the Diffusion equations for the specified system (intrinsic or extraction mode)

		:returns: None
		"""

		self.fit_parameters_log = []

		
		if self.propagation_method == 'exp':
			self.propagate_ode_exp()
		elif self.propagation_method == 'integrate':
			self.propagate_ode_integrate()
		else:
			print("Error, propagation method not recognized, program quitting")
			exit()


	def make_animation(self,gif_path, save_flag):

		population_trace = self.pop[:,0]
		time_trace = np.linspace(0,self.t_max, self.nT)
		len_trace = np.linspace(0,self.L, self.nL)

		print(self.pop.shape)
		print(time_trace.shape)

		fig = plt.figure()
		ax = fig.add_subplot(121)
		self.plot_obj, = ax.plot(len_trace, population_trace)
		ax.set(ylim=[0,np.amax(self.pop)*0.8], title='Spatial distribution at time ')
		ax2 = fig.add_subplot(122)
		ax2.plot(time_trace, np.sum(self.pop, axis=0))
		ax2.set(ylim=[0,np.amax( np.sum(self.pop, axis=0))*1.2], title='Total population vs. time')

		def update(i):
			new_pop_trace = self.pop[:,i]
			self.plot_obj.set_data(len_trace, new_pop_trace)
			return (self.plot_obj,)

		anim = matplotlib.animation.FuncAnimation(fig, update, frames=self.nT, interval=12, blit=True)
		plt.show()
		if save_flag:
			anim.save(Path(self.output_path,gif_path))
		# anim.save(Path(self.output_path,gif_path), fps=30, extra_args=['-vcodec', 'libx264'])



class program_tester(object):
	"""
	class defining a tester object, used to analyze the behavior of the DiffusionSystem objects (debug)
	"""
	def __init__(self,program_object):
		"""
		assigns a DiffusionSystem object to the tester

		:param program_object: DiffusionSystem object to be tested
		:type program_object: DiffusionSystem

		:returns: None
		"""
		self.program = program_object

	def check_initialization(self):
		"""
		prints to terminal the values of the initialization variables

		:returns: None
		"""
		print("-- Spatial grid")
		print("\nFilm thickness: {}\nNumber of grid points: {}\nGrid step-size: {}\n".format(self.program.L, self.program.nL, self.program.dL))
		print("-- Time points")
		print("\nTotal evolution time: {}\nNumber of time points: {}\nTime step-size: {}\n".format(self.program.t_max, self.program.nT, self.program.dT))
		print("-- Film properties")
		print("\nDecay lifetime: {}\nDiffusion coefficient: {}\n".format(self.program.tau, self.program.De))
		print("Film mobility (cm^2/(V*s)): {}\nExtraction-layer mobility (cm^2/(V*s)): {}\n".format(self.program.film_e_mobility*1e4, self.program.extracted_e_mobility*1e4))
		print("\nAbsorbance: {}\nAbsorption coefficient: {}\n".format(self.program.absorbance, self.program.alpha))
		print("-- Methods")
		print("\nTime-propagation method: {}".format(self.program.propagation_method))



if __name__ == "__main__":
	# program class instantation
	program = DiffusionSystem()

	# initialization
	program.initialize_spatial_grid(100e-9, 500)	# 1st: film thickness in nm / 2nd: number of sptial grid points in the z-direction
	program.initialize_maxT(1000e-12)					# duration of the simulation, in s
	program.initialize_dT(1e-12)					# time-step of the simulation, in s
	program.initialize_time_grid()					# computes the number of time steps
	program.initialize_temperature(300)				# temperature in K
	program.initialize_relative_electric_charge(1)		# relative electric charge of the mobile ion
	
	program.initialize_mode("drift_diffusion")
	program.initialize_propagation_method("integrate")

	# NOTE: comment this block out if you are using "drift_diffusion_varying_cs" mode, uncomment it if you are using "drift_diffusion"
	program.initialize_mobility_film(1)	# mobility in the InP QD film, in cm^2/(V*s), converted in m^2/(V*s)
	program.calculate_diffusion_coefficient() # diffusion coefficient in m^2/s
	program.initialize_electric_field(3e6)			# units: V/m
    # NOTE: block's end

	## NOTE: comment this block out if you are using "drift_diffusion" mode, uncomment it if you are using "drift_diffusion_varying_cs"
	# program.initialize_mobility_film_steps([0.2,0.2,1,1])	# mobility in the InP QD film, in cm^2/(V*s), converted in m^2/(V*s)
	# program.initialize_electric_field_steps([0,0,0,0])
	# # program.initialize_electric_field_constant(0)
	# program.calculate_diffusion_coefficient_vector() # diffusion coefficient in m^2/s
    ## NOTE: block's end

	# set program output
	program.set_output('./output')
	
	print("-- Initialization concluded successfully")

	# show initial results
	print("-- Solving diffusion system")
	program.solve_DiffusionSystem()
	print("-- System solved, quitting")

	program.make_animation("diffusion_only_test.gif", save_flag=False)
