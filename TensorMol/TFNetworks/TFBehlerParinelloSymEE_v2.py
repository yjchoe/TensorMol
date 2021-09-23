"""
A TF2 implementation of the BPNN network using the tf.keras API.

Focus is porting the model implementation into tf.keras.
Further work would include streamlining data iteration using tf.data,
    generalizing to other models, etc.
"""


from .TFInstance import *
from ..Containers.TensorMolData import *
from .TFMolInstance import *
from ..ForceModels.ElectrostaticsTF import *
from ..ForceModifiers.Neighbors import *
from ..TFDescriptors.RawSymFunc import *

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import time

assert tf.__version__.startswith("2"), (
    f"TensorFlow version 2+ required, got {tf.__version__}"
)

# Need eager
tf.compat.v1.enable_eager_execution()
LOGGER.info("[tf2] Using tf2 with the tf.keras API in %s mode",
            "eager" if tf.executing_eagerly() else "graph")


class MolInstance_DirectBP_EandG_SymFunction(MolInstance_fc_sqdiff_BP):
    """
    Behler Parinello Scheme with energy and gradient training.
    NO Electrostatic embedding.

    **Uses TF2's `tf.keras` module
    without sessions, (explicitly defined) graphs or placeholders.**
    """

    def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ",
                 share_weights=False):
        """
        Args:
            TData_: A TensorMolData instance.
            Name_: A name for this instance.
            Trainable_: True for training, False for evalution
            ForceType_: Deprecated
            share_weights: Whether to share weights between input elements
        """
        self.SFPa = None
        self.SFPr = None
        self.Ra_cut = None
        self.Rr_cut = None
        self.HasANI1PARAMS = False
        MolInstance.__init__(self, TData_, Name_, Trainable_)
        self.MaxNAtoms = self.TData.MaxNAtoms
        self.eles = self.TData.eles
        self.n_eles = len(self.eles)
        self.eles_np = np.asarray(self.eles).reshape((self.n_eles, 1))
        self.eles_pairs = []
        for i in range(len(self.eles)):
            for j in range(i, len(self.eles)):
                self.eles_pairs.append([self.eles[i], self.eles[j]])
        self.eles_pairs_np = np.asarray(self.eles_pairs)
        LOGGER.info("HasANI1PARAMS, %s", self.HasANI1PARAMS)
        if not self.HasANI1PARAMS:
            LOGGER.info("calling SetANI1Param")
            self.SetANI1Param()
        self.HiddenLayers = PARAMS["HiddenLayers"]
        self.batch_size = PARAMS["batch_size"]
        self.share_weights = share_weights
        LOGGER.info("HiddenLayers: %s", self.HiddenLayers)
        LOGGER.info("activation_function_type: %s",
                    self.activation_function_type)
        LOGGER.info("share_weights: %s", self.share_weights)
        if (self.Trainable):
            self.TData.LoadDataToScratch(self.tformer)
        self.summary_writer = None
        self.learning_rate = PARAMS["learning_rate"]
        self.weight_decay = PARAMS.get("weight_decay", 0.001)
        self.suffix = PARAMS["NetNameSuffix"]
        self.SetANI1Param()

        self.GradScalar = PARAMS["GradScalar"]
        self.EnergyScalar = PARAMS["EnergyScalar"]
        self.TData.ele = self.eles_np
        self.TData.elep = self.eles_pairs_np

        self.NetType = "RawBP_EandG"
        self.name = "Mol_" + self.TData.name + "_" + self.TData.dig.name + "_" + self.NetType + "_" + self.suffix
        self.train_dir = PARAMS["networks_directory"] + self.name
        self.keep_prob = np.asarray(PARAMS["KeepProb"])
        self.nlayer = len(PARAMS["KeepProb"]) - 1
        self.monitor_mset = PARAMS["MonitorSet"]
        self.chk_start = 0  # kan
        self.training = None

        # TF2 update: prepare tf.Tensor versions of the ANI1 parameters
        self.tf_int = tf.int64
        self.eles_tf = tf.convert_to_tensor(self.eles_np, dtype=self.tf_int)
        self.eles_pairs_tf = tf.convert_to_tensor(self.eles_pairs_np, dtype=self.tf_int)
        self.SFPa2_tf = tf.convert_to_tensor(self.SFPa2, dtype=self.tf_prec)
        self.SFPr2_tf = tf.convert_to_tensor(self.SFPr2, dtype=self.tf_prec)
        self.Rr_cut_tf = tf.convert_to_tensor(self.Rr_cut, dtype=self.tf_prec)
        self.Ra_cut_tf = tf.convert_to_tensor(self.Ra_cut, dtype=self.tf_prec)
        self.zeta_tf = tf.convert_to_tensor(self.zeta, dtype=self.tf_prec)
        self.eta_tf = tf.convert_to_tensor(self.eta, dtype=self.tf_prec)
        # input order to TFSymSet_*
        self.symset_params = (
            self.eles_tf, self.SFPr2_tf, self.Rr_cut_tf, self.eles_pairs_tf,
            self.SFPa2_tf, self.zeta_tf, self.eta_tf, self.Ra_cut_tf,
        )
        # lazily build tf.keras modules (see TrainPrepare and energy_inference)
        self.model = None
        self.optim = None

    def SetANI1Param(self, prec=np.float64):
        """
        Generate ANI1 symmetry function parameter tensor.
        """
        self.Ra_cut = PARAMS["AN1_a_Rc"]
        self.Rr_cut = PARAMS["AN1_r_Rc"]
        zetas = np.array([[PARAMS["AN1_zeta"]]], dtype=prec)
        etas = np.array([[PARAMS["AN1_eta"]]], dtype=prec)
        AN1_num_a_As = PARAMS["AN1_num_a_As"]
        AN1_num_a_Rs = PARAMS["AN1_num_a_Rs"]
        thetas = np.array([2.0 * Pi * i / AN1_num_a_As for i in range(0, AN1_num_a_As)], dtype=prec)
        rs = np.array([self.Ra_cut * i / AN1_num_a_Rs for i in range(0, AN1_num_a_Rs)], dtype=prec)
        # Create a parameter tensor. 4 x nzeta X neta X ntheta X nr
        p1 = np.tile(np.reshape(zetas, [1, 1, 1, 1, 1]), [1, 1, AN1_num_a_As, AN1_num_a_Rs, 1])
        p2 = np.tile(np.reshape(etas, [1, 1, 1, 1, 1]), [1, 1, AN1_num_a_As, AN1_num_a_Rs, 1])
        p3 = np.tile(np.reshape(thetas, [1, 1, AN1_num_a_As, 1, 1]), [1, 1, 1, AN1_num_a_Rs, 1])
        p4 = np.tile(np.reshape(rs, [1, 1, 1, AN1_num_a_Rs, 1]), [1, 1, AN1_num_a_As, 1, 1])
        SFPa = np.concatenate([p1, p2, p3, p4], axis=4)
        self.SFPa = np.transpose(SFPa, [4, 0, 1, 2, 3])
        etas_R = np.array([[PARAMS["AN1_eta"]]], dtype=prec)
        AN1_num_r_Rs = PARAMS["AN1_num_r_Rs"]
        rs_R = np.array([self.Rr_cut * i / AN1_num_r_Rs for i in range(0, AN1_num_r_Rs)], dtype=prec)
        # Create a parameter tensor. 2 x  neta X nr
        p1_R = np.tile(np.reshape(etas_R, [1, 1, 1]), [1, AN1_num_r_Rs, 1])
        p2_R = np.tile(np.reshape(rs_R, [1, AN1_num_r_Rs, 1]), [1, 1, 1])
        SFPr = np.concatenate([p1_R, p2_R], axis=2)
        self.SFPr = np.transpose(SFPr, [2, 0, 1])
        self.inshape = int(len(self.eles) * AN1_num_r_Rs + len(self.eles_pairs) * AN1_num_a_Rs * AN1_num_a_As)
        self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
        # self.inshape = int(len(self.eles) * AN1_num_r_Rs)
        p1 = np.tile(np.reshape(thetas, [AN1_num_a_As, 1, 1]), [1, AN1_num_a_Rs, 1])
        p2 = np.tile(np.reshape(rs, [1, AN1_num_a_Rs, 1]), [AN1_num_a_As, 1, 1])
        SFPa2 = np.concatenate([p1, p2], axis=2)
        self.SFPa2 = np.transpose(SFPa2, [2, 0, 1])
        p1_new = np.reshape(rs_R, [AN1_num_r_Rs, 1])
        self.SFPr2 = np.transpose(p1_new, [1, 0])
        self.zeta = PARAMS["AN1_zeta"]
        self.eta = PARAMS["AN1_eta"]
        self.HasANI1PARAMS = True
        LOGGER.info("SetANI1Param inshape: %s", self.inshape)

    def Clean(self):
        """
        Clean Instance for pickle saving.

        *TF2 update: many of the v1 attributes (sess, train_op, ...) removed.*
        """
        self.model = None
        self.inputs = None
        self.outputs = None
        self.optim = None

        self.eles_tf = None
        self.eles_pairs_tf = None
        self.SFPa2_tf = None
        self.SFPr2_tf = None
        self.Rr_cut_tf = None
        self.Ra_cut_tf = None
        self.zeta_tf = None
        self.eta_tf = None

    def make_neighbor_layer(self, xyzs, Zs, natoms):
        """Encode neighborhood info for input features within tf.keras.Model.

        NOTE: this is a TF function wrapper for the numpy routine.
        Gradients will not go through this routine.
        """

        def _build_pairs_np(xyzs, Zs, natoms, Rr_cut, Ra_cut, eles, eles_pairs):
            NL = NeighborListSet(
                xyzs, natoms,
                DoTriples_=True, DoPerms_=True, ele_=Zs, sort_=True,
            )
            return NL.buildPairsAndTriplesWithEleIndexLinear(
                Rr_cut, Ra_cut, eles, eles_pairs)

        def _build_pairs_tf(args):
            """Wrapper of numpy_function for the Lambda layer.

            Reference:
                https://github.com/tensorflow/tensorflow/issues/34674
            """
            return tf.numpy_function(_build_pairs_np, args, [self.tf_int] * 4)

        Radp_Ele, Angt_Elep, mil_j, mil_jk = \
            tf.keras.layers.Lambda(_build_pairs_tf)([
                xyzs, Zs, natoms,
                self.Rr_cut_tf, self.Ra_cut_tf, self.eles_tf, self.eles_pairs_tf
            ])

        return Radp_Ele, Angt_Elep, mil_j, mil_jk

    def prepare_batch(self, xyzs, Zs,
                      target_energy=None, target_grads=None,
                      Radp_Ele=None, Angt_Elep=None, mil_j=None, mil_jk=None,
                      natoms_inv=None):
        """Prepares a raw numpy input batch for a tf2 model.

        Input batch is given by `MolInstance.GetTrainBatch(batch_size)`.
        """

        xyzs = tf.convert_to_tensor(xyzs, dtype=self.tf_prec)
        Zs = tf.convert_to_tensor(Zs, dtype=self.tf_int)
        if target_energy is not None:
            target_energy = tf.convert_to_tensor(target_energy,
                                                 dtype=self.tf_prec)
            target_grads = tf.convert_to_tensor(target_grads,
                                                dtype=self.tf_prec)

        if Radp_Ele is not None:
            Radp_Ele = tf.convert_to_tensor(Radp_Ele, dtype=self.tf_int)
            Angt_Elep = tf.convert_to_tensor(Angt_Elep, dtype=self.tf_int)
            mil_j = tf.convert_to_tensor(mil_j, dtype=self.tf_int)
            mil_jk = tf.convert_to_tensor(mil_jk, dtype=self.tf_int)

        # shouldn't really be None in default cases
        if natoms_inv is not None:
            natoms_inv = tf.convert_to_tensor(natoms_inv, dtype=self.tf_prec)
            natoms = tf.cast(1. / natoms_inv, dtype=self.tf_int)

        return (
            xyzs, Zs, natoms,
            target_energy, target_grads,
            Radp_Ele, Angt_Elep, mil_j, mil_jk, natoms_inv,
        )

    def TrainPrepare(self, continue_training=False):
        """Setup tf.keras.Model for training."""
        self.training = True
        return self.Prepare(continue_training=continue_training)

    def EvalPrepare(self, instance_name=None):
        """Setup tf.keras.Model for evaluation.

        A trained model is retrieved from
            `os.path.join(PARAMS["networks_directory"], instance_name)`.
        """
        self.training = False
        return self.Prepare(instance_name=instance_name)

    def Prepare(self, continue_training=False, instance_name=None):
        """
        Setup tf.keras.Model for training or evaluation.

        *TF2 update: returns a tf.keras.Model.
        BPNN's inputs are now just `[xyzs, Zs, natoms]`, and
        it includes the symmetry functions computation within the model.
        Loss functions are defined directly in the training step.
        Finally, model save/load routines are replaced with
        `tf.keras.Model.save_weights()` and `tf.keras.Model.load_weights()`.*
        """

        # Network inputs (placeholder equivalents; 1st dimension is batch size)
        xyzs = tf.keras.Input(
            shape=(self.MaxNAtoms, 3), dtype=self.tf_prec, name="InputCoords")
        Zs = tf.keras.Input(
            shape=(self.MaxNAtoms, ), dtype=self.tf_int, name="InputElems")
        natoms = tf.keras.Input(
            shape=(), dtype=self.tf_int, name="InputNAtoms")
        self.inputs = (xyzs, Zs, natoms)

        # BPNN
        # (i) symmetry functions
        Radp_Ele, Angt_Elep, mil_j, mil_jk = \
            self.make_neighbor_layer(*self.inputs)
        scatter_sym, sym_index = TFSymSet_Scattered_Linear_WithEle_Release(
            xyzs, Zs,
            *self.symset_params,
            Radp_Ele, Angt_Elep, mil_j, mil_jk,
        )
        scatter_sym = [tf.ensure_shape(t, shape=(None, self.inshape))
                       for t in scatter_sym]
        sym_index = [tf.ensure_shape(t, shape=(None, 2))
                     for t in sym_index]
        # (ii) atomic neural networks
        Etotal, Ebp, Ebp_atom = self.energy_inference(
            scatter_sym, sym_index, xyzs, self.keep_prob,
        )
        # (iii) combined energy (only Etotal is used for training)
        self.outputs = (Etotal, Ebp, Ebp_atom)

        # tf.keras.Model
        self.model = tf.keras.Model(
            inputs=self.inputs,
            outputs=self.outputs,
            name="fc_sqdiff_BP_Direct_EandG_SymFunction",
        )
        LOGGER.info(f"Trainable parameters: %d",
                    np.sum([np.prod(t.get_shape().as_list())
                            for t in self.model.trainable_weights]))

        # tf.keras.Optimizer
        if self.training:
            self.optim = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            LOGGER.info(f"using Adam optimizer with lr {self.learning_rate}")

        # tf.train.Checkpoint (simplified)
        if instance_name is not None:
            ckpt_dir = os.path.join(PARAMS["networks_directory"],
                                    instance_name)
        else:
            ckpt_dir = self.train_dir
        if not self.training or continue_training:
            ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if ckpt is not None:
                LOGGER.info(f"restoring latest checkpoint from %s", ckpt)
                self.model.load_weights(ckpt)
            elif self.training:
                LOGGER.info(
                    f"no saved checkpoint found in %s, training from scratch",
                    ckpt_dir,
                )
            else:
                raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")
        else:
            LOGGER.info("training from scratch, saving to %s", self.train_dir)

    def energy_inference(self, inp, indexs, xyzs, keep_prob):
        """
        Builds a Behler-Parinello graph for calculating energy.

        *TF2 update: define using tf.keras layers*

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
            xyzs: xyz coordinates of atoms.
            keep_prob: dropout prob of each layer.
        Returns:
            The BP graph energy output
        """
        # BPNN
        def _make_layer(n_in, n_out, l2_wd=0.001, name=None):
            """Make a tf.keras dense layer with a specific initializer."""
            return tf.keras.layers.Dense(
                n_out,
                activation=self.activation_function_type,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0.,
                    stddev=1. / (10 + np.sqrt(n_in)),
                ),  # could just use HeNormal or VarianceScaling?
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.L2(l2=l2_wd),
                name=name,
            )

        layer_sizes = [self.inshape] + self.HiddenLayers

        # TF2 change: don't use external batch size!
        batch_size = tf.shape(xyzs)[0]
        outputs = tf.zeros((batch_size, self.MaxNAtoms), dtype=self.tf_prec)

        if self.share_weights:
            hidden_layers = [
                _make_layer(n_in, n_out,
                            l2_wd=self.weight_decay, name=f"hidden{j}")
                for j, (n_in, n_out) in enumerate(
                    zip(layer_sizes[:-1], layer_sizes[1:]))
            ]
            output_layer = _make_layer(layer_sizes[-1], 1,
                                       l2_wd=0.0,
                                       name="regression_linear")
        else:
            hidden_layers = []
            output_layer = None

        # MLP for each element
        for i, elem in enumerate(self.eles):
            h = inp[i]
            if self.share_weights:
                for hidden_layer, keep_p in zip(hidden_layers, keep_prob):
                    h = hidden_layer(h)
                    if keep_p < 1.0:
                        h = tf.keras.layers.Dropout(1. - keep_p)(h)
            else:
                for j, (n_in, n_out, keep_p) in enumerate(
                        zip(layer_sizes[:-1], layer_sizes[1:], keep_prob)):
                    hidden_layer = _make_layer(
                        n_in, n_out,
                        l2_wd=self.weight_decay,
                        name=f"elem{elem:02d}_hidden{j}",
                    )
                    h = hidden_layer(h)
                    if keep_p < 1.0:
                        h = tf.keras.layers.Dropout(1. - keep_p)(h)
                output_layer = _make_layer(
                    layer_sizes[-1], 1,
                    l2_wd=0.0,
                    name=f"elem{elem:02d}_regression_linear",
                )
            out = output_layer(h)  # (flattened_batch, 1)
            natoms_batch = tf.shape(out)[0]  # scalar tensor

            atom_index = indexs[i][:, 1:]
            atom_energy = tf.reshape(
                tf.scatter_nd(atom_index,
                              tf.reshape(out, (natoms_batch, )),
                              (batch_size * self.MaxNAtoms, )),
                (batch_size, self.MaxNAtoms),
            )
            tf.debugging.assert_all_finite(
                atom_energy,
                f"nan encountered in output for {elem}",
            )
            outputs += atom_energy

        bp_energy = tf.reshape(tf.reduce_sum(outputs, axis=1),
                               (batch_size, ))

        total_energy = tf.identity(bp_energy)
        return total_energy, bp_energy, outputs

    def loss_fn(self, energy, grads, target_energy, target_grads, natoms_inv):
        """
        loss function that includes dipole loss, energy loss and gradient loss.

        Inputs: Etotal, gradient, Elabels, grads, 1 / natom_inv

        *TF2 update: replaces loss_op.*
        """
        max_natoms = tf.cast(tf.shape(grads)[2], self.tf_prec)
        energy_loss = tf.nn.l2_loss(
            (target_energy - energy) * natoms_inv * max_natoms
        )
        grads_loss = tf.nn.l2_loss(
            (target_grads - grads) * tf.reshape(natoms_inv * max_natoms,
                                                [tf.shape(grads)[0], 1, 1])
        )
        EandG_loss = (
            self.EnergyScalar * energy_loss + self.GradScalar * grads_loss
        )
        return EandG_loss, energy_loss, grads_loss

    def run_epoch(self, epoch, training=True, profiling=False):
        """Run through the training or the test set for one epoch."""

        assert self.model is not None, (
            "tf.keras.Model is not initialized. Call self.TrainPrepare() first"
        )
        assert training or not profiling, (
            "profiling can only be done on training data"
        )

        if training or profiling:
            n_cases = self.TData.NTrain
            batch_fn = lambda ncases: self.TData.GetTrainBatch(ncases, False)
        else:
            n_cases = self.TData.NTest
            batch_fn = lambda ncases: self.TData.GetTestBatch(ncases, False)

        total_loss = 0.0
        total_energy_loss = 0.0
        total_grads_loss = 0.0
        pbar = tf.keras.utils.Progbar(
            int(n_cases / self.batch_size) * self.batch_size,
            stateful_metrics=["loss", "energy_loss", "grads_loss"])

        if training and not profiling:
            LOGGER.info(f"[Epoch {epoch}]")
        else:
            LOGGER.info(f"[Evaluation after epoch {epoch}]")
        for ministep in range(0, int(n_cases / self.batch_size)):
            (xyzs, Zs, natoms, target_energy, target_grads,
             _, _, _, _, natoms_inv) = \
                self.prepare_batch(*batch_fn(self.batch_size))

            if profiling:
                t0 = time.time()

            # TF2 training routine w/ gradient computations
            with tf.GradientTape() as tape, tf.GradientTape() as egrad_tape:
                egrad_tape.watch(xyzs)
                energy, _, _ = self.model((xyzs, Zs, natoms), training=training)
                energy_grads = egrad_tape.gradient(energy, xyzs)
                loss, energy_loss, grads_loss = self.loss_fn(
                    energy, energy_grads, target_energy, target_grads,
                    natoms_inv,
                )

            if profiling:
                LOGGER.info("inference time", time.time() - t0)
                self.summary_writer.add_run_metadata(
                    self.run_metadata, 'ministep%d' % ministep)
                fetched_timeline = timeline.Timeline(
                    self.run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_step_%d.json' % ministep, 'w') as f:
                    f.write(chrome_trace)

            if training:
                self.optim.minimize(loss, self.model.trainable_weights,
                                    tape=tape)

            losses = {
                "loss": loss.numpy().item(),
                "energy_loss": energy_loss.numpy().item(),
                "grads_loss": grads_loss.numpy().item(),
            }
            pbar.add(
                self.batch_size,
                values=[(name, l / self.batch_size)
                        for name, l in losses.items()],
            )
            total_loss += losses["loss"]
            total_energy_loss += losses["energy_loss"]
            total_grads_loss += losses["grads_loss"]

        n = int(n_cases / self.batch_size) * self.batch_size
        loss = total_loss / n
        energy_loss = total_energy_loss / n
        grads_loss = total_grads_loss / n
        mode = ("profiling" if profiling else
                "training" if training else "testing")
        LOGGER.debug("%s loss %.5f energy_loss %.5f grads_loss %.5f",
                     mode, loss, energy_loss, grads_loss)
        return loss, energy_loss, grads_loss

    def train_step(self, epoch):
        """
        Perform a single training epoch (complete processing of all input),
        using minibatches of size self.batch_size.
        Training object including dipole, energy and gradient

        Args:
            epoch: the index of this step.
        """
        return self.run_epoch(epoch, training=True)

    def test(self, epoch):
        """
        Perform a single test epoch (complete processing of all input),
        using minibatches of size self.batch_size

        Args:
            epoch: the index of this step.
        """
        return self.run_epoch(epoch, training=False)

    def save_chk(self, step):
        """Save model checkpoint at step.

        *TF2 update: replaces session saver with `tf.keras.Model.save_weights`.*
        """
        self.chk_file = os.path.join(self.train_dir, f"cp-{step:04d}.ckpt")
        LOGGER.info("saving model checkpoint %s", self.chk_file)
        self.model.save_weights(self.chk_file)

    def train(self, mxsteps, continue_training=False):
        """
        Iterate over train_step and test.

        A 'step' is really an 'epoch'.
        """
        self.TrainPrepare(continue_training)
        test_freq = PARAMS["test_freq"]
        best_loss = float('inf')
        # start = self.global_step.eval() # kan get last global_step
        start = self.chk_start + 1
        if (start >= mxsteps):
            LOGGER.info("max steps %d exceeds current checkpoint index %d",
                        mxsteps, start)
            return
        for step in range(start, mxsteps + 1):
            self.train_step(step)
            if step % test_freq == 0 and step != 0:
                if self.monitor_mset is not None:
                    self.InTrainEval(self.monitor_mset,
                                     self.Rr_cut, self.Ra_cut, step=step)
                LOGGER.debug("train calling test")
                test_loss, _, _ = self.test(step)
                if test_loss < best_loss:
                    best_loss = test_loss
                    LOGGER.debug("train calling save_chk")
                    self.save_chk(step)
        # LOGGER.debug("train calling SaveAndClose")
        # self.SaveAndClose()

    def profile_step(self, epoch):
        """
        Perform a single profiling epoch (complete processing of all input),
        using minibatches of size self.batch_size

        Args:
            epoch: the index of this epoch.
        """
        return self.run_epoch(step, training=True, profiling=True)

    def profile(self):
        """
        This profiles a training step.
        """
        LOGGER.info("[Profiling %s]", self.name)
        self.TrainPrepare(False)
        self.profile_step(1)
        return

    def InTrainEval(self, mol_set, Rr_cut, Ra_cut, step=0):
        """
        Evalute the network during training.
        """
        LOGGER.info("caculating energy on the monitor set")

        nmols = len(mol_set.mols)
        for i in range(nmols, self.batch_size):
            mol_set.mols.append(mol_set.mols[-1])
        nmols = len(mol_set.mols)
        xyzs = np.zeros((nmols, self.MaxNAtoms, 3), dtype=np.float64)
        Zs = np.zeros((nmols, self.MaxNAtoms), dtype=np.int32)
        natoms = np.zeros((nmols), dtype=np.int32)
        for i, mol in enumerate(mol_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            natoms[i] = mol.NAtoms()

        # build neighbors
        NL = NeighborListSet(
            xyzs, natoms,
            DoTriples_=True, DoPerms_=True, ele_=Zs, sort_=True,
        )
        Radp_Ele, Angt_Elep, mil_j, mil_jk = \
            NL.buildPairsAndTriplesWithEleIndexLinear(
                Rr_cut, Ra_cut, self.eles_np, self.eles_pairs_np)

        # convert to tensors
        batch = self.prepare_batch(xyzs, Zs, natoms_inv=1.0 / natoms)
        xyzs, Zs, natoms = batch[:3]

        # forward model
        with tf.GradientTape() as tape:
            tape.watch(xyzs)
            Etotal, Ebp, Ebp_atom = self.model((xyzs, Zs, natoms),
                                               training=False)
        Etotal_grad = tape.gradient(Etotal, xyzs)

        monitor_data = [Etotal, Ebp, Ebp_atom, Etotal_grad]
        with open(self.name + "_monitor_" + str(step) + ".dat", "wb") as f:
            pickle.dump(monitor_data, f)
        return Etotal, Ebp, Ebp_atom, Etotal_grad

    def evaluate_e(self, xyzs, Zs, natoms_inv):
        """
        Evaluate the Behler-Parinello energy, atom energies

        *TF2 Update: inputs & forward (greatly) simplified*
        """
        batch = self.prepare_batch(xyzs, Zs, natoms_inv=natoms_inv)
        xyzs, Zs, natoms = batch[:3]
        with tf.GradientTape() as tape:
            tape.watch(xyzs)
            Etotal, Ebp, Ebp_atom = self.model((xyzs, Zs, natoms),
                                               training=False)
        Etotal_grad = tape.gradient(Etotal, xyzs)
        return Etotal, Ebp, Ebp_atom, Etotal_grad

    def EvalBPDirectEandGLinear(self, mol, Rr_cut, Ra_cut,Set=True):
        """
        The energy, force and dipole routine for BPs_EE set or single mol.

        *TF2 Update: moved from manager to instance.*
        """
        if Set:
            mol_set = mol
            self.TData.MaxNAtoms = mol_set.MaxNAtoms()
        else:
            mol_set=MSet()
            mol_set.mols.append(mol)
            self.TData.MaxNAtoms = mol.NAtoms()
        nmols = len(mol_set.mols)
        dummy_energy = np.zeros((nmols))
        xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype=np.float64)
        dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3),
                               dtype=np.float64)
        Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype=np.int32)
        natoms = np.zeros((nmols), dtype=np.int32)
        for i, mol in enumerate(mol_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            natoms[i] = mol.NAtoms()

        LOGGER.debug('EvalBPDirectEandGLinear gets E and G from evaluate_e')
        Etotal, Ebp, Ebp_atom, gradient = \
            self.evaluate_e(xyzs, Zs, 1.0 / natoms)
        return Etotal, Ebp, Ebp_atom, -JOULEPERHARTREE * gradient[0]

    def EvalBPDirectEandGLinearSingle(self, mol, Rr_cut, Ra_cut):
        """
        The energy, force and dipole routine for BPs_EE.
        """
        mol_set = MSet()
        mol_set.mols.append(mol)
        nmols = len(mol_set.mols)
        dummy_energy = np.zeros((nmols))
        self.TData.MaxNAtoms = mol.NAtoms()
        xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype=np.float64)
        dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3),
                               dtype=np.float64)
        Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype=np.int32)
        natoms = np.zeros((nmols), dtype=np.int32)
        for i, mol in enumerate(mol_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            natoms[i] = mol.NAtoms()

        Etotal, Ebp, Ebp_atom, gradient = \
            self.evaluate_e(xyzs, Zs, 1.0 / natoms)
        return Etotal, Ebp, Ebp_atom, -JOULEPERHARTREE * gradient[0]
