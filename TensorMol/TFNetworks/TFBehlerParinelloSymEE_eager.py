"""
These instances are re-writes of the convoluted instances found in TFMolInstanceDirect.

I would still like the following changes:
- Independence from any manager.
- Inheritance from a re-written instance base class.
- Removal of any dependence on TensorMolData
- Removal of any dependence on TFInstance.

But at least these are a first step.  JAP 12/2017.
KAN:
    testing tf.enable_eager_execution with def evaluate_e
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

TF2_MODE = False
if sys.version_info[0] < 3:
	import tensorflow as tf
	tf.compat.v1.enable_eager_execution()
else:
	if(TF2_MODE):
		import tensorflow as tf
		print("Using Tensorflow version "+tf.__version__)
	else:
		import tensorflow.compat.v1 as tf # use tf2 with py3
		tf.compat.v1.enable_eager_execution()
		print("Using Tensorflow version "+tf.__version__+" with compat.v1.enable_eager_execution ")

from .TFInstance import *
from ..Containers.TensorMolData import *
from .TFMolInstance import *
from ..ForceModels.ElectrostaticsTF import *
from ..ForceModifiers.Neighbors import *
from ..TFDescriptors.RawSymFunc import *
from tensorflow.python.client import timeline
import time
import threading

class MolInstance_DirectBP_EandG_SymFunction(MolInstance_fc_sqdiff_BP):
	"""
	Behler Parinello Scheme with energy and gradient training.
	NO Electrostatic embedding.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
			Trainable_: True for training, False for evalution
			ForceType_: Deprecated
		"""
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.HasANI1PARAMS = False
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		print ("MolInstance_DirectBP_EandG_SymFunction: self.HasANI1PARAMS ",self.HasANI1PARAMS)
		if not self.HasANI1PARAMS:
			print ("MolInstance_DirectBP_EandG_SymFunction: call SetANI1Param ")
			self.SetANI1Param()
		self.HiddenLayers = PARAMS["HiddenLayers"]
		self.batch_size = PARAMS["batch_size"]
		print ("MolInstance_DirectBP_EandG_SymFunction:self.HiddenLayers: ", self.HiddenLayers)
		print ("self.activation_function_type: ", self.activation_function_type)
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.grads_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.learning_rate = PARAMS["learning_rate"]
		self.suffix = PARAMS["NetNameSuffix"]
		self.SetANI1Param()
		self.run_metadata = None

		self.GradScalar = PARAMS["GradScalar"]
		self.EnergyScalar = PARAMS["EnergyScalar"]
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np

		self.NetType = "RawBP_EandG"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+self.suffix
		self.train_dir = PARAMS["networks_directory"]+self.name
		self.keep_prob = np.asarray(PARAMS["KeepProb"])
		self.nlayer = len(PARAMS["KeepProb"]) - 1
		self.monitor_mset =  PARAMS["MonitorSet"]
		self.chk_start = 0 # kan

	def SetANI1Param(self, prec=np.float64):
		"""
		Generate ANI1 symmetry function parameter tensor.
		"""
		self.Ra_cut = PARAMS["AN1_a_Rc"]
		self.Rr_cut = PARAMS["AN1_r_Rc"]
		zetas = np.array([[PARAMS["AN1_zeta"]]], dtype = prec)
		etas = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
		AN1_num_a_As = PARAMS["AN1_num_a_As"]
		AN1_num_a_Rs = PARAMS["AN1_num_a_Rs"]
		thetas = np.array([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype = prec)
		rs =  np.array([ self.Ra_cut*i/AN1_num_a_Rs for i in range (0, AN1_num_a_Rs)], dtype = prec)
		# Create a parameter tensor. 4 x nzeta X neta X ntheta X nr
		p1 = np.tile(np.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p3 = np.tile(np.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_a_Rs,1])
		p4 = np.tile(np.reshape(rs,[1,1,1,AN1_num_a_Rs,1]),[1,1,AN1_num_a_As,1,1])
		SFPa = np.concatenate([p1,p2,p3,p4],axis=4)
		self.SFPa = np.transpose(SFPa, [4,0,1,2,3])
		etas_R = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
		AN1_num_r_Rs = PARAMS["AN1_num_r_Rs"]
		rs_R =  np.array([ self.Rr_cut*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype = prec)
		# Create a parameter tensor. 2 x  neta X nr
		p1_R = np.tile(np.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
		p2_R = np.tile(np.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
		SFPr = np.concatenate([p1_R,p2_R],axis=2)
		self.SFPr = np.transpose(SFPr, [2,0,1])
		self.inshape = int(len(self.eles)*AN1_num_r_Rs + len(self.eles_pairs)*AN1_num_a_Rs*AN1_num_a_As)
		self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
		#self.inshape = int(len(self.eles)*AN1_num_r_Rs)
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]
		self.HasANI1PARAMS = True
		print("MolInstance_DirectBP_EandG_SymFunction:SetANI1Param: self.inshape:", self.inshape)
		print("Number of AN1 parameters: self.inshape:", self.inshape)

	def Clean(self):
		"""
		Clean Instance for pickle saving.
		"""
		Instance.Clean(self)
		#self.tf_prec = None
		self.xyzs_pl, self.Zs_pl, self.label_pl, self.grads_pl, self.natom_pl = None, None, None, None, None
		self.check, self.options, self.run_metadata = None, None, None
		self.atom_outputs = None
		self.energy_loss = None
		self.Scatter_Sym, self.Sym_Index = None, None
		self.Radp_pl, self.Angt_pl = None, None
		self.Elabel_pl = None
		self.Etotal, self.Ebp, self.Ebp_atom = None, None, None
		self.gradient = None
		self.total_loss_dipole, self.energy_loss, self.grads_loss = None, None, None
		self.train_op_dipole, self.train_op_EandG = None, None
		self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG = None, None, None, None
		self.Radp_Ele_pl, self.Angt_Elep_pl = None, None
		self.mil_jk_pl, self.mil_j_pl = None, None
		self.keep_prob_pl = None
		return


	def TrainPrepare(self,  continue_training =False):
		"""
		Define Tensorflow graph for training.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.Etotal, self.gradient, self.Elabel_pl, self.grads_pl, self.natom_pl)
			tf.summary.scalar("loss", self.loss)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
			config.gpu_options.per_process_gpu_memory_fraction = 0.90
			self.sess = tf.Session(config=config)
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			ckpt = tf.train.get_checkpoint_state(self.train_dir) # kan
			self.sess.run(init)
			if ckpt and ckpt.model_checkpoint_path:          
				print(" kan continue_training")
				print(ckpt.model_checkpoint_path)
				chkfile = ckpt.model_checkpoint_path
				#chknum = int(chkfile.replace(self.name+'-chk-','')))
				chknum = chkfile.replace(self.name+'-chk-','')
				print("chknum ", chknum) # kan
				chknum2 = chknum.replace(self.name,'')
				print("chknum2 ", chknum2) # kan
				self.chk_start = int(chknum2.replace('./networks//',''))
				#self.saver.restore(self.sess, self.chk_file)
				self.saver.restore(self.sess, ckpt.model_checkpoint_path)
				print("Start from:", self.chk_start) # kan
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			raise Exception("Please check your inputs")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.mil_j_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict

	def energy_inference(self, inp, indexs, xyzs, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating energy.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph energy output
		"""
		
		#tf.print("zw: self.Scatter_Sym",inp,summarize=-1)   # same as before calling e_inference
		#tf.print("zw: self.xyzs_pl",xyzs,summarize=-1)
		#print("zw: self all",self)
		#print("zw: tf.trainable_variables=",tf.compat.v1.trainable_variables())

		
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				inputs = inp[e]
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							if (self.chk_file): tf.train.Checkpoint(weights=weights,biases=biases).restore(self.chk_file) #zw
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))

							#print("zw i,weights_np ",i,weights.numpy())
							#print("zw i,biases_np ",i,biases.numpy())
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							if (self.chk_file): tf.train.Checkpoint(weights=weights,biases=biases).restore(self.chk_file) #zw
							#self.saver = tf.contrib.eager.Saver(var_list=[weights,biases]) #zw error
							#self.saver = tf.train.Checkpoint(weights=weights,biases=biases) #zw
							#self.saver.restore(self.chk_file) #zw
							
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
							#print("zw i,weights ",i,weights)
							#print("zw i,biases ",i,biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					if (self.chk_file): tf.train.Checkpoint(weights=weights,biases=biases).restore(self.chk_file) #zw
					Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Ebranches[-1][-1])
					cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
					#print("zw ToAdd arg ",atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms],[self.batch_size, self.MaxNAtoms])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					#print("zw ToAdd ",ToAdd)
					#print("zw output ",output)
					output = tf.add(output, ToAdd)
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
			bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])
		total_energy = tf.identity(bp_energy)
		#print("kan energy_inference: total_energy",total_energy)
		#print_op=tf.print("kan energy_inference: total_energy",total_energy)
		#tf.control_dependencies([print_op])
		return total_energy, bp_energy, output


	def loss_op(self, energy, energy_grads, Elabels, grads, natom):
		"""
		losss function that includes dipole loss, energy loss and gradient loss.
		"""
		maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
		energy_diff  = tf.multiply(tf.subtract(energy, Elabels,name="EnDiff"), natom*maxatom)
		energy_loss = tf.nn.l2_loss(energy_diff,name="EnL2")
		grads_diff = tf.multiply(tf.subtract(energy_grads, grads,name="GradDiff"), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
		grads_loss = tf.nn.l2_loss(grads_diff,name="GradL2")
		EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar),name="MulLoss")
		loss = tf.identity(EandG_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss

	def training(self, loss, learning_rate, momentum):
		"""Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.
		Args:
		loss: Loss tensor, from loss().
		learning_rate: The learning rate to use for gradient descent.
		Returns:
		train_op: The Op for training.
		"""
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam")
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step, name="trainop")
		return train_op

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size.
		Training object including dipole, energy and gradient

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss, Etotal = self.sess.run([self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.Etotal], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, num_of_mols, duration)
		return

	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		test_grads_loss = 0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size) + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss, Etotal = self.sess.run([self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.Etotal], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		#print ("testing... in test ")
		print ("kan testing... in test (TFBehlerParinelloSymEE.py) ")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, num_of_mols, duration, False)
		return test_loss

	def train(self, mxsteps, continue_training= False):
		"""
		This the training loop for the united model.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		LOGGER.info("in class MolInstance_DirectBP_EandG_SymFunction (kan)") #kan
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_dipole_test_loss = float('inf') # some big numbers
		mini_energy_test_loss = float('inf')
		mini_test_loss = float('inf')
		#start = self.global_step.eval() # kan get last global_step
		start = self.chk_start + 1
		if(start >= mxsteps):
			print('Increase max_steps', mxsteps)
			return
		#print("Start from:", start) # kan
		#for step in  range (0, mxsteps): # kan
		for step in  range (start, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				if self.monitor_mset != None:
					self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, step=step)
				print("kan train calls test")
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					print("kan train calls save_chk")
					self.save_chk(step)
		print("kan train calls SaveAndClose")
		self.SaveAndClose()
		return


	def profile_step(self, step):
		"""
		Perform a single profiling step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		time_print_mini = time.time()
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, Etotal = self.sess.run([self.train_op, self.Etotal], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			print ("inference time:", time.time() - t)
			self.summary_writer.add_run_metadata(self.run_metadata, 'minstep%d' % ministep)
			duration = time.time() - start_time
			num_of_mols += actual_mols
			fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_step_%d.json' % ministep, 'w') as f:
				f.write(chrome_trace)
		return

	def profile(self):
		"""
		This profiles a training step.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(False)
		self.profile_step(1)
		return

	def InTrainEval(self, mol_set, Rr_cut, Ra_cut, step=0):
		"""
		Evaluted the network during training.
		"""
		nmols = len(mol_set.mols)
		for i in range(nmols, self.batch_size):
			mol_set.mols.append(mol_set.mols[-1])
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		xyzs = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexLinear(Rr_cut, Ra_cut, self.eles_np, self.eles_pairs_np)
		batch_data = [xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, mil_j, mil_jk, 1.0/natom]
		feed_dict=self.fill_feed_dict(batch_data + [np.ones(self.nlayer+1)])
		Etotal, Ebp, Ebp_atom, gradient= self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.gradient], feed_dict=feed_dict)
		monitor_data = [Etotal, Ebp, Ebp_atom, gradient]
		f = open(self.name+"_monitor_"+str(step)+".dat","wb")
		pickle.dump(monitor_data, f)
		f.close()
		print ("calculating monitoring set..")
		return Etotal, Ebp, Ebp_atom, gradient

	def print_training(self, step, loss, energy_loss, grads_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("kan step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		else:
			LOGGER.info("kan step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		return

	def evaluate_e(self, batch_data):
		"""
		Evaluate the Behler-Parinello energy, atom energies
		"""
		print ("########## TFBehlerParinelloSymEE_eager ###########")
		print ("##### MolInstance_DirectBP_EandG_SymFunction:evaluate_e: ######")
		nmol = batch_data[2].shape[0]
		self.activation_function_type = PARAMS["NeuronType"]
		self.AssignActivation()
		#print ("self.activation_function:\n\n", self.activation_function)
		#print ("self.batch_size:", self.batch_size, " nmol:", nmol)
		#print ("batch_data:", batch_data[0]) # coordinates for all molecules in batch
		if (batch_data[0].shape[1] != self.MaxNAtoms or self.batch_size != nmol):
			self.MaxNAtoms = batch_data[0].shape[1]
			self.batch_size = nmol
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
		#print ('self.eles_n ', self.eles_np,'self.eles_pairs_np ',self.eles_pairs_np)
		#print ('self.Rr_cut',self.Rr_cut)
		#print ('self.SFPa2',self.SFPa2)
		#print ('self.SFPr2',self.SFPr2)
		self.batch_size = nmol
		Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
		Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
		SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
		SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
		Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
		#Rr_cut = self.Rr_cut
		Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
		#Ra_cut = self.Ra_cut
		#print ("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.Ra_cut:", self.Ra_cut)
		zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
		#zeta = self.zeta
		#print ("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.zeta:", self.zeta)
		eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
		#eta = self.eta
		#print ("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.eta:", self.eta)
		#self.xyzs_pl=batch_data[0] # coordinates for all molecules in batch
		#print ("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: batch_data[0]", batch_data[0].shape)
		self.xyzs_pl=tf.Variable(batch_data[0],trainable=False, dtype = self.tf_prec,name="InputCoords")
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.xyzs_pl",self.xyzs_pl) # kan
		#self.Zs_pl = batch_data[1]
		self.Zs_pl = tf.Variable(batch_data[1],trainable=False, dtype =tf.int64, name="InputZs")
		#self.Radp_Ele_pl=batch_data[4]
		self.Radp_Ele_pl=tf.Variable(batch_data[4],trainable=False,dtype=tf.int64,)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.Radp_Ele_pl",self.Radp_Ele_pl)
		#self.Angt_Elep_pl=batch_data[5]
		self.Angt_Elep_pl=tf.Variable(batch_data[5],trainable=False,dtype=tf.int64)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.Angt_Elep_pl",self.Angt_Elep_pl)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.Angt_Elep_pl",self.Angt_Elep_pl.shape)
		#self.mil_j_pl = batch_data[6]
		self.mil_j_pl = tf.Variable(batch_data[6],trainable=False,dtype=tf.int64)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.mil_jk_pl",self.mil_jk_pl)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.mil_jk_pl",self.mil_jk_pl.shape)
		#self.mil_jk_pl = batch_data[7]
		self.mil_jk_pl = tf.Variable(batch_data[7],trainable=False,dtype=tf.int64)
		self.natom_pl = tf.Variable(self.batch_size,trainable=False,dtype=self.tf_prec)
		#self.keep_prob_pl = PARAMS["KeepProb"]
		self.keep_prob_pl = tf.Variable(PARAMS["KeepProb"],trainable=False,dtype=self.tf_prec)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e: self.keep_prob_pl",self.keep_prob_pl)
		#print("MolInstance_DirectBP_EandG_SymFunction:evaluate_e calls TFSymSet_Scattered_Linear_WithEle_Release")
		#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
		#sys.exit(2)
		#print("EvalPrepare calls energy_inference")
		#tf.print("zw: self.Scatter_Sym",self.Scatter_Sym,summarize=-1)
		#print("zw: self.xyzs_pl",self.xyzs_pl)
		#tf.print("zw: tf: self.xyzs_pl",self.xyzs_pl,summarize=-1)
		#print("zw: self.Sym_Index",self.Sym_Index)
		#print("zw: self.keep_prob_pl",self.keep_prob_pl)

		#self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
		#self.Etotal, self.Ebp, self.Ebp_atom, self.gradient = self.EandG_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
		#zw: calc gradient
		with tf.GradientTape(persistent = True,watch_accessed_variables=False) as tape:
			tape.watch(self.xyzs_pl)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
		self.gradient  = tape.gradient(self.Etotal, self.xyzs_pl)
		#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
		#self.gradient  = tf.GradientTape().gradient(self.Etotal, self.xyzs_pl, name="BPEGrad")
		#self.gradient  = tf.GradientTape(persistent=True).gradient(self.Etotal, self.xyzs_pl)
		#dummy_grads = np.zeros((nmol, self.MaxNAtoms, 3), dtype = np.float64)
		#self.gradient  = dummy_grads # need to work on gradient tf.GradientTape instead
		#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
		#self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
		print("evaluate_e: self.Etotal",self.Etotal)
		print("evaluate_e, self.gradient",self.gradient)
		#tf.print("evaluate_e, tf self.gradient",self.gradient[0],summarize=-1)
		print ("kan  MolInstance_DirectBP_EandG_SymFunction:evaluate_e self.Etotal", self.Etotal)
		#sys.exit(2)
		return self.Etotal, self.Ebp, self.Ebp_atom, self.gradient

