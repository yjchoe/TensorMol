"""
Routines for running external Ab-Initio packages to get shit out of mol.py
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random, math, subprocess
# kan from pyscf import gto, scf, dft
from TensorMol import * # kan for unit conversions
def PyscfDft(m_,basis_ = '6-31g*',xc_='b3lyp'):
	#if (not HAS_PYSCF):
	#	print("Missing PYSCF")
	#	return 0.0
        #print('enter pyscdft')
	mol = gto.Mole()
	pyscfatomstring=""
	crds = m_.coords.copy()
	crds[abs(crds)<0.0001] *=0.0
	for j in range(len(m_.atoms)):
		#pyscfatomstring=pyscfatomstring+str(m_.atoms[j])+" "+str(crds[j,0])+" "+str(crds[j,1])+" "+str(crds[j,2])+(";" if j!= len(m_.atoms)-1 else "")
		pyscfatomstring=pyscfatomstring+str(m_.atoms[j])+" "+str(crds[j,0])+" "+str(crds[j,1])+" "+str(crds[j,2])+("\n " if j!= len(m_.atoms)-1 else "")
	mol.atom = pyscfatomstring
        #print(mol.atom)
	mol.unit = "Angstrom"
	mol.charge = 0
	mol.spin = 0
	mol.basis = basis_
	#mol.verbose = 0
	mol.verbose = 5
	mol.build()
        #dft.numint.libxc = dft.xcfun # kan
	mf = dft.RKS(mol)
	mf.xc = xc_
	e = mf.kernel()
	# add gradients and dipoles (KAN 08/2018)
	grd = mf.nuc_grad_method().kernel()
	dip=mf.dip_moment(verbose=0)
	return e,grd*JOULEPERHARTREE*BOHRPERA # for md 
	#return e,grd*JOULEPERHARTREE*BOHRPERA,dip
def G09DFT(m_,basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', filename_='tmp', path_='./g09/', joule_per_a=True,mem=False,nproc=4,nprocl=False):
	#istring = '%chk='+filename_+'.chk\n'
	istring = ''
	if mem:
	    istring = istring+'%mem='+str(mem)+'\n'
	else:
	    istring = istring+'%mem=1500Mb\n'
	if nprocl:
	    istring = istring+'%nprocl='+str(nprocl)+'\n'
	if nproc:
	    istring = istring+'%nproc='+str(nproc)+'\n'
	else:
	    istring = istring+'%nproc=4\n'
	if jobtype_ == 'sp':
	    #istring = istring+'#p '+xc_+','+basis_+',scf=(tight,direct),pop=Hirshfeld\n'
	    istring = istring+'#p '+xc_+','+basis_+',int=(Grid=Ultrafine),scf=(tight,direct)\n'
	else:
	    istring = istring+'#p '+xc_+','+basis_+','+jobtype_+',int=(Grid=Ultrafine),scf=(tight,direct)\n'
	istring = istring+'\n#c\n' # comment card
	istring = istring+'\n0 1 \n' # charge and mult
	crds = m_.coords.copy()
	crds[abs(crds)<0.0000] *=0.0
	for j in range(len(m_.atoms)):
		istring=istring+str(m_.atoms[j])+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
	istring=istring+'\n'
	with open(path_+filename_+'.com','w') as fin:
		fin.write(istring)
	if nprocl:
	   proc = subprocess.Popen(['g09l_e01', path_+filename_], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
	else:
	   proc = subprocess.Popen(['g09_e01', path_+filename_], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
	#out, err = proc.communicate()
	proc.communicate() # output comming back
	with open(path_+filename_+'.log','r') as f:
		out = f.read()
	lines = out.split('\n')
	if jobtype_ == 'force':
		Forces = np.zeros((m_.atoms.shape[0],3))
		#print(Forces)
		for i, line in enumerate(lines):
			if line.count('SCF Done:')>0:
				Energy = float(line.split()[4])
				#print(Energy)
			if "Dipole moment (field-independent basis, Debye):" in line:
				tmp = lines[i+1].split()
				dipole = np.asarray([float(tmp[1]),float(tmp[3]),float(tmp[5])])
				#print(dipole)
			if "Forces (Hartrees/Bohr)" in line:
				for j in range(len(m_.atoms)):
					tmp = lines[i+3+j].split()
					Forces[j,:] = float(tmp[2]), float(tmp[3]),float(tmp[4])          
		#print(Forces)
		if(joule_per_a):
			return Energy, Forces*JOULEPERHARTREE*BOHRPERA, dipole
		else:
			return Energy, Forces, dipole
	elif jobtype_ == 'sp':
		for i, line in enumerate(lines):
			if line.count('SCF Done:')>0:
				Energy = float(line.split()[4])
				print(Energy)
			if "Dipole moment (field-independent basis, Debye):" in line:
				tmp = lines[i+1].split()
				dipole = np.asarray([float(tmp[1]),float(tmp[3]),float(tmp[5])])
				print(dipole)
		return Energy, dipole
	else:
		raise Exception("jobtype needs formatted for return variables")


def QchemDFT(m_,basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', filename_='tmp', path_='./qchem/', threads=False):
	istring = '$molecule\n0 1 \n'
	crds = m_.coords.copy()
	crds[abs(crds)<0.0000] *=0.0
	for j in range(len(m_.atoms)):
		#kanistring=istring+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
		istring=istring+str(m_.atoms[j])+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
	if jobtype_ == "dipole":
		istring =istring + '$end\n\n$rem\njobtype sp\nbasis '+basis_+'\nmethod '+xc_+'\nthresh 11\nsymmetry false\nsym_ignore true\n$end\n'
	else:
		istring =istring + '$end\n\n$rem\njobtype '+jobtype_+'\nbasis '+basis_+'\nmethod '+xc_+'\nthresh 11\nUNRESTRICTED   true\nsymmetry false\nsym_ignore true\n$end\n'
	with open(path_+filename_+'.in','w') as fin:
		fin.write(istring)
	with open(path_+filename_+'.out','a') as fout:
		if threads:
			proc = subprocess.Popen(['qchem', '-nt', str(threads), path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		else:
			proc = subprocess.Popen(['qchem', path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		out, err = proc.communicate()
		fout.write(out)
	lines = out.split('\n')
	if jobtype_ == 'force':
		Forces = np.zeros((m_.atoms.shape[0],3))
		for i, line in enumerate(lines):
			if line.count('Convergence criterion met')>0:
				Energy = float(line.split()[1])
			if line.count("Gradient of SCF Energy") > 0:
				k = 0
				l = 0
				for j in range(1, m_.atoms.shape[0]+1):
					Forces[j-1,:] = float(lines[i+k+2].split()[l+1]), float(lines[i+k+3].split()[l+1]), float(lines[i+k+4].split()[l+1])
					l += 1
					if (j % 6) == 0:
						k += 4
						l = 0
		return Energy, -Forces*JOULEPERHARTREE*BOHRPERA
	elif jobtype_ == 'sp':
		for line in lines:
			if line.count('Convergence criterion met')>0:
				Energy = float(line.split()[1])
		return Energy
	elif jobtype_ ==  'dipole':
		for i, line in enumerate(lines):
			if "Dipole Moment (Debye)" in line:
				tmp = lines[i+1].split()
				dipole = np.asarray([float(tmp[1]),float(tmp[3]),float(tmp[5])])
				return dipole
	else:
		raise Exception("jobtype needs formatted for return variables")

def QchemDFT_optimize(m_,basis_ = '6-31g*',xc_='b3lyp', filename_='tmp', path_='./qchem/', threads=False):
	istring = '$molecule\n0 1 \n'
	crds = m_.coords.copy()
	crds[abs(crds)<0.0000] *=0.0
	atoms = m_.atoms.shape[0]
	for j in range(len(m_.atoms)):
		istring=istring+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
	istring =istring + '$end\n\n$rem\njobtype opt\ngeom_opt_max_cycles 500\nbasis '+basis_+'\nmethod '+xc_+'\nthresh 11\nsymmetry false\nsym_ignore true\n$end\n'
	with open(path_+filename_+'.in','w') as fin:
		fin.write(istring)
	with open(path_+filename_+'.out','a') as fout:
		if threads:
			proc = subprocess.Popen(['qchem', '-nt', str(threads), path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		else:
			proc = subprocess.Popen(['qchem', path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		out, err = proc.communicate()
		fout.write(out)
	lines = out.split('\n')
	for i, line in enumerate(lines):
		if "**  OPTIMIZATION CONVERGED  **" in line:
			with open(path_+filename_+'.xyz','a') as fopt:
				xyz = []
				for j in range(atoms):
					xyz.append([lines[i+5+j].split()[1],lines[i+5+j].split()[2],lines[i+5+j].split()[3],lines[i+5+j].split()[4]])
				fopt.write(str(atoms)+"\nComment:\n")
				for j in range(atoms):
					fopt.write(xyz[j][0]+"      "+xyz[j][1]+"      "+xyz[j][2]+"      "+xyz[j][3]+"\n")
				fopt.write("\n")
			break

def QchemRIMP2(m_,basis_ = 'cc-pvtz', aux_basis_='rimp2-cc-pvtz', jobtype_='force', filename_='tmp', path_='./qchem/', threads=False):
	istring = '$molecule\n0 1 \n'
	crds = m_.coords.copy()
	crds[abs(crds)<0.0000] *=0.0
	for j in range(len(m_.atoms)):
		istring=istring+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
	istring =istring + '$end\n\n$rem\njobtype '+jobtype_+'\nbasis '+basis_+'\nAUX_BASIS '+aux_basis_+'\nmethod rimp2\nsymmetry false\nsym_ignore true\nmem_total 9999\nmem_static 2000\n$end\n'
	#print istring
	with open(path_+filename_+'.in','w') as fin:
		fin.write(istring)
	with open(path_+filename_+'.out','a') as fout:
		if threads:
			proc = subprocess.Popen(['qchem', '-nt', str(threads), path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		else:
			proc = subprocess.Popen(['qchem', path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
		out, err = proc.communicate()
		fout.write(out)
	lines = out.split('\n')
	if jobtype_ == 'force':
		Forces = np.zeros((m_.atoms.shape[0],3))
		for i, line in enumerate(lines):
			if line.count('RI-MP2 TOTAL ENERGY')>0:
				Energy = float(line.split()[4])
			if line.count("Full Analytical Gradient of MP2 Energy") > 0:
				k = 0
				l = 0
				for j in range(1, m_.atoms.shape[0]+1):
					Forces[j-1,:] = float(lines[i+k+2].split()[l+1]), float(lines[i+k+3].split()[l+1]), float(lines[i+k+4].split()[l+1])
					l += 1
					if (j % 5) == 0:
						k += 4
						l = 0
		# return Energy, Forces
		return -Forces*JOULEPERHARTREE/BOHRPERA
	elif jobtype_ == 'sp':
		for line in lines:
			if line.count('RI-MP2 TOTAL ENERGY')>0:
				Energy = float(line.split()[4])
		return Energy
	else:
		raise Exception("jobtype needs formatted for return variables")


def PullFreqData():
	a = open("/media/sdb1/dtoth/qchem_jobs/new/phenol.out", "r+") #Change file name
	# each time to read correct output file
	f=open("phenol_freq.dat", "w") #Change file name to whatever you want --
	# make sure it's different each time
	lines = a.readlines()
	data = []
	ip = 0
	for i, line in enumerate(lines):
		if line.count("NAtoms") > 0:
			atoms = int(lines[i+1].split()[0])
			break
	nm = np.zeros((3*atoms-6, atoms, 3))
	for i, line in enumerate(lines):
		if "Frequency:" in line:
			freq = [line.split()[1], line.split()[2],line.split()[3]]
			intens = [lines[i+4].split()[2], lines[i+4].split()[3],lines[i+4].split()[4]]
			f.write(freq[0] + "   " + intens[0] + "\n")
			f.write(freq[1] + "   " + intens[1] + "\n")
			f.write(freq[2] + "   " + intens[2] + "\n")
		if "Raman Active" in line:
			for j in range(atoms):
				it = 0
				for k in range(3):
					for l in range(3):
						nm[it+ip,j,l] = float(lines[i+j+2].split()[k*3+l+1])
					it += 1
			ip += 3
			# f.write(nm[0] + "  " + nm)
	np.save("morphine_nm.npy", nm)
	f.close()

def PySCFMP2Energy(m, basis_='cc-pvqz'):
	mol = gto.Mole()
	pyscfatomstring=""
	for j in range(len(m.atoms)):
		s = m.coords[j]
		pyscfatomstring=pyscfatomstring+str(m.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(m.atoms)-1 else "")
	mol.atom = pyscfatomstring
	mol.basis = basis_
	mol.verbose = 0
	try:
		mol.build()
		mf=scf.RHF(mol)
		hf_en = mf.kernel()
		mp2 = mp.MP2(mf)
		mp2_en = mp2.kernel()
		en = hf_en + mp2_en[0]
		m.properties["energy"] = en
		return en
	except Exception as Ex:
		print("PYSCF Calculation error... :",Ex)
		print("Mol.atom:", mol.atom)
		print("Pyscf string:", pyscfatomstring)
		return 0.0
		#raise Ex
	return
