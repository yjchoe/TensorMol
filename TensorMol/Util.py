from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gc, random, os, sys, re, atexit
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	#kan import _pickle as pickle
	import pickle as pickle
import random, math, time, itertools, warnings
from math import pi as Pi
import scipy.special
#from collections import defaultdict
#from collections import Counter
from TensorMol.TMParams import *
TMBanner()
from TensorMol.PhysicalData import *
warnings.simplefilter(action = "ignore", category = FutureWarning)
#
# GLOBALS
#	Any global variables of the code must be put here, and must be in all caps.
#	Global variables are almost never acceptable except in these few cases

# PARAMETERS
PARAMS = TMParams()
#
# -- begin Environment set up.
#
# if any of these paths do not exist create them.
def MakeDirIfAbsent(path):
	if sys.version_info[0]+sys.version_info[1]*0.1<3.2:
		try:
			os.makedirs(path)
		except OSError:
			if not os.path.isdir(path):
				raise
	else:
		os.makedirs(path, exist_ok=True)
MakeDirIfAbsent(PARAMS["sets_dir"])
MakeDirIfAbsent(PARAMS["networks_directory"])
MakeDirIfAbsent(PARAMS["log_dir"])
MakeDirIfAbsent(PARAMS["results_dir"])
MakeDirIfAbsent(PARAMS["dens_dir"])
MakeDirIfAbsent(PARAMS["log_dir"])
LOGGER = TMLogger(PARAMS["log_dir"])
MAX_ATOMIC_NUMBER = 10
# Derived Quantities and useful things.
#  TODO: Migrate these to PARAMS
HAS_PYSCF = False
HAS_EMB = False
HAS_TF = False
GRIDS = None
HAS_GRIDS=False

LOGGER.info("Searching for Installed Optional Packages...")
try:
	from pyscf import scf
	from pyscf import gto
	from pyscf import dft
	from pyscf import mp
	HAS_PYSCF = True
	LOGGER.debug("Pyscf has been found")
except Exception as Ex:
	LOGGER.info("Pyscf is not installed -- no ab-initio sampling")
	pass

try:
	import MolEmb
	HAS_EMB = True
	LOGGER.debug("MolEmb has been found.")

except Exception as Ex:
	print("MolEmb is not installed. Please cd C_API; sudo python setup.py install",Ex)
	pass

#kan try:
	#import tensorflow as tf
	#LOGGER.debug("Tensorflow version "+tf.__version__+" has been found")
	#HAS_TF = True
#except:
	#LOGGER.info("Tensorflow not Installed, very limited functionality")
	#pass

try:
	import multiprocessing
	N_CORES=multiprocessing.cpu_count()
	LOGGER.debug("Found "+str(N_CORES)+" CPUs to thread over... ")
except:
	LOGGER.info("Only a single CPU, :( did you lose a war?")
	pass
LOGGER.debug("TensorMol ready...")

LOGGER.debug("TMPARAMS----------")
LOGGER.debug(PARAMS)
LOGGER.debug("TMPARAMS~~~~~~~~~~")

if (HAS_PYSCF and HAS_GRIDS):
	from TensorMol.Grids import *
	GRIDS = Grids()
	GRIDS.Populate()
print("--------------------------")
#
# -- end Environment set up.
#

# A simple timing decorator.
TMTIMER = {}
TMSTARTTIME = time.time()
def PrintTMTIMER():
	LOGGER.info("=======    Accumulated Time Information    =======")
	LOGGER.info("Category   |||   Time Per Call   |||   Total Elapsed     ")
	for key in TMTIMER.keys():
		if (TMTIMER[key][1]>0):
			LOGGER.info(key+" ||| %0.5f ||| %0.5f ",TMTIMER[key][0]/(TMTIMER[key][1]),TMTIMER[key][0])
def TMTiming(nm_="Obs"):
	if (not nm_ in TMTIMER.keys()):
		TMTIMER[nm_] = [0.,0]
	def wrap(f):
		def wf(*args,**kwargs):
			t0 = time.time()
			output = f(*args,**kwargs)
			TMTIMER[nm_][0] += time.time()-t0
			TMTIMER[nm_][1] += 1
			return output
		LOGGER.debug("TMTimed "+nm_+str(TMTIMER[nm_]))
		return wf
	return wrap

@atexit.register
def exitTensorMol():
	PrintTMTIMER()
	LOGGER.info("Total Time : %0.5f s",time.time()-TMSTARTTIME)
	LOGGER.info("~ Adios Homeshake ~")

#
# All the garbage below here needs to be removed, organized
# and the perpretrators genetalia scattered to the four winds.
#

def complement(a,b):
	return [i for i in a if b.count(i)==0]

def scitodeci(sci):
	tmp=re.search(r'(\d+\.?\d+)\*\^(-?\d+)',sci)
	return float(tmp.group(1))*pow(10,float(tmp.group(2)))

def AtomicNumber(Symb):
	try:
		return atoi[Symb]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def AtomicSymbol(number):
	try:
		return atoi.keys()[atoi.values().index(number)]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def LtoS(l):
	s=""
	for i in l:
		s+=str(i)+" "
	return s

def nCr(n, r):
	f = math.factorial
	return int(f(n)/f(r)/f(n-r))

# Wow... Holy shit kun. Stop putting stuff here. Totally inappropriate.

def DSF(R, R_c, alpha):	# http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581 damp shifted force
	if R > R_c:
		return 0.0
	else:
		twooversqrtpi = 1.1283791671
		XX = alpha*R_c
		ZZ = scipy.special.erfc(XX)/R_c
		YY = twooversqrtpi*alpha*math.exp(-XX*XX)/R_c
		LR = (scipy.special.erfc(alpha*R)/R - ZZ + (R-R_c)*(ZZ/R_c+YY))
		return LR

def DSF_Gradient(R, R_c, alpha):
	if R > R_c:
		return 0.0
	else:
		twooversqrtpi = 1.1283791671
		XX = alpha*R_c
		ZZ = scipy.special.erfc(XX)/R_c
		YY = twooversqrtpi*alpha*math.exp(-XX*XX)/R_c
		grads = -((scipy.special.erfc(alpha*R)/R/R + twooversqrtpi*alpha*math.exp(-alpha*R*alpha*R)/R)-(ZZ/R_c + YY))
		return grads

def EluAjust(x, a, x0, shift):
	if x > x0:
		return a*(x-x0)+shift
	else:
		return a*(math.exp(x-x0)-1.0)+shift

def sigmoid_with_param(x, prec=tf.float64):
	return tf.log(1.0+tf.exp(tf.multiply(tf.cast(PARAMS["sigmoid_alpha"], dtype=prec), x)))/tf.cast(PARAMS["sigmoid_alpha"], dtype=prec)

def guassian_act(x, prec=tf.float64):
	return tf.exp(-x*x)

def guassian_rev_tozero(x, prec=tf.float64):
	return tf.where(tf.greater(x, 0.0), 1.0-tf.exp(-x*x), tf.zeros_like(x))

def guassian_rev_tozero_tolinear(x, prec=tf.float64):
	a = 0.5
	b = -0.06469509698101589
	x0 = 0.2687204431537632
	step1 = tf.where(tf.greater(x, 0.0), 1.0-tf.exp(-x*x), tf.zeros_like(x))
	return tf.where(tf.greater(x, x0), a*x+b, step1)

def square_tozero_tolinear(x, prec=tf.float64):
	a = 1.0
	b = -0.0025
	x0 = 0.005
	step1 = tf.where(tf.greater(x, 0.0), 100.0*x*x, tf.zeros_like(x))
	return tf.where(tf.greater(x, x0), a*x+b, step1)
