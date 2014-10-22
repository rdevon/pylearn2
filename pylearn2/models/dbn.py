import numpy as np
import warnings

from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.models import mlp
from pylearn2.models import Model
from pylearn2.models.dbm import layer as dbm_layer
from pylearn2.models.dbm import RBM
from pylearn2.models.dbm.inference_procedure import InferenceProcedure
from pylearn2.models.dbm.sampling_procedure import SamplingProcedure
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.extra_ops import fill_diagonal

class DBN(Model):
    """
    Deep belief network.

    Parameters
    ----------
    lower_model : DBN or RBM
        DBN is built out of a DBN or RBM. These will be used for the forward inference.
    rbm : RBM
        The top rbm for the model. Used for generative learning.
    forward_inference_procedure: string
        "SAMPLING" or "MF". Indicates the method used for forward inference.
    theano_rng: TODO
    """

    # We use this dicitonary to map rbm visible layers to hidden layer.
    visible_hidden_dict = {
        dbm_layer.BinaryVector: dbm_layer.BinaryVectorMaxPool
    }

    @staticmethod
    def check_layers(lower_hidden_layer, upper_visible_layer):
        if isinstance(lower_hidden_layer, dbm_layer.BinaryVectorMaxPool) and lower_hidden_layer.pool_size != 1:
            raise NotImplementedError("%r does not support pooling layers yet" % type(self))

        if lower_hidden_layer.detector_layer_dim != upper_visible_layer.nvis:
            raise ValueError("Dimension of lower RBM hidden layer"
                             "do not match visible layer of top RBM (%d vs %d)"
                             % (lower_hidden_layer.detector_layer_dim, upper_visible_layer.nvis))

        if type(lower_hidden_layer) != DBN.visible_hidden_dict[type(upper_visible_layer)]:
            raise ValueError("Type mismatch of lower RBM hidden layer"
                             " and visible layer of top RBM (%r vs %r)"
                             % (type(lower_hidden_layer), type(upper_visible_layer)))

    @staticmethod
    def match_layers(lower_hidden, upper_visible):
        DBN.check_layers(lower_hidden, upper_visible)
        if isinstance(lower_hidden, dbm_layer.BinaryVectorMaxPool):
            upper_visible.center = lower_hidden.center
            upper_visible.set_biases(lower_hidden.get_biases())
            if upper_visible.center:
                upper_visible.offset = sharedX(sigmoid_numpy(upper_visible.bias.get_value()))
        else:
            raise NotImplementedError("Cannot yet handle %r hidden with %r visible"
                                                         % (type(lower_hidden), type(upper_visible)))

    def __init__(self, lower_model, rbm,
                        forward_method=None,
                        theano_rng=None):

        self.forward_method = forward_method

        # The other RBMs are used to derive the forward pass MLP as well as get reconstructions.
        if isinstance(lower_model, RBM):
            self.rbms = [lower_model] + [rbm]
            self.input_space = lower_model.visible_layer.space
        elif isinstance(lower_model, DBN):
            self.rbms = lower_model.rbms + [rbm]
            self.input_space = lower_model.input_space
        else:
            raise ValueError("lower model must be RBM or DBN, not %r" % type(lower_model))
        # Check RBM structure
        for i in range(len(self.rbms) - 1):
            DBN.match_layers(self.rbms[i].hidden_layers[0], self.rbms[i+1].visible_layer)

        # Set the top RBM for inference / sampling procedures / etc
        self.top_rbm = self.rbms[-1]

        self.force_batch_size = self.top_rbm.batch_size

        # We need a theano_rng to do sampling for forward inference.
        if theano_rng is None:
            self.theano_rng = MRG_RandomStreams(2014 + 10 + 17)
        else:
            self.theano_rng = theano_rng

        # Set batch sizes  and rename layers.
        for i, rbm in enumerate(self.rbms):
            rbm.set_batch_size(self.top_rbm.batch_size)
            rbm.visible_layer.layer_name = "dbn_rbm_%d_v" % i
            rbm.hidden_layers[0].layer_name = "dbn_rbm_%d_h" % i

        # Create an mlp with corresponding layers and shared variables as parameters.
        mlp_layers = []
        for i, rbm in enumerate(self.rbms):
            mlp_layers.append(mlp.PretrainedLayer("mlp_layer_%d" % i,
                rbm.hidden_layers[0].make_shared_mlp_layer()))
        self.mlp = mlp.MLP(mlp_layers,
                                        batch_size=self.top_rbm.batch_size)

        # DBN needs some extra members from RBM
        self.visible_layer = self.top_rbm.visible_layer
        self.hidden_layers = self.top_rbm.hidden_layers

        self.inference_procedure = DBN_Inference_Procedure()
        self.inference_procedure.dbn = self
        self.sampling_procedure = DBN_Sampling_Procedure()
        self.sampling_procedure.dbn = self

    def feed_forward(self, X, Y=None, method=None, level=None, theano_rng=None):
        """
        DBN needs to feed forward through the layers first before working with the top.
        """

        if self.forward_method is not None:
            method = self.forward_method

        if method not in ["MF", "SAMPLING"]:
            raise ValueError("Forward methods supported are MF or SAMPLING, not %r" % method)

        if theano_rng is None:
            theano_rng = self.theano_rng

        state_below = None
        for i, rbm in enumerate(self.rbms[:-1]):
            if level is not None and i > level:
                break
            if i == 0:
                state_below = rbm.visible_layer.upward_state(X)
            else:
                assert state_below is not None

            if method == "MF":
                state = rbm.hidden_layers[0].mf_update(state_below, state_above=None)
            elif method == "SAMPLING":
                state = rbm.hidden_layers[0].sample(state_below, theano_rng=theano_rng)
            state_below = rbm.hidden_layer.upward_state(state)

        return state_below

    def generate(self, source_rbm, state, to_level=0):

        rbms = []
        for rbm in self.rbms:
            rbms.append(rbm)
            if rbm == source_rbm:
                break
            if rbm == self.rbms[-1]:
                raise ValueError("Source RBM %r not found DBN rbms" % rbm)

        assert len(rbms) - to_level > 0
        rbms = rbms[to_level:]

        for i, rbm in enumerate(rbms[::-1]):
            layer_above = rbm.hidden_layers[0]
            downward_state=state
            state = rbm.visible_layer.inpaint_update(
                layer_above=layer_above,
                state_above=downward_state,
                drop_mask=None, V=None)

        return state

    def get_sampling_updates(self, layer_to_state, theano_rng,
                                               layer_to_clamp=None, num_steps=1,
                                               return_layer_to_updated=False):
        return self.top_rbm.get_sampling_updates(layer_to_state, theano_rng,
                                                               layer_to_clamp=layer_to_clamp,
                                                               num_steps=num_steps,
                                                               return_layer_to_updated=return_layer_to_updated)

    def energy(self, X, hidden):
        return self.top_rbm.energy(X, hidden)

    def expected_energy(self, X, mf_hidden):
        X = self.feed_forward(X, method="MF")
        return self.top_rbm.expected_energy(X, mf_hidden)

    def mf(self, *args, **kwargs):
        return self.inference_procedure.mf(*args, **kwargs)

    def make_layer_to_state(self, num_examples, rng=None):
        return self.top_rbm.make_layer_to_state(num_examples, rng=rng)

    def make_layer_to_symbolic_state(self, num_examples, rng=None):
        return self.top_rbm.make_layer_to_symbolic_state(num_examples, rng=rng)

    def get_params(self):
        return self.top_rbm.get_params()

    def initialize_chains(self, X, Y, theano_rng):
        X_hat = self.feed_forward(X, method="SAMPLING")
        return self.top_rbm.initialize_chains(X_hat, Y, theano_rng)

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channel.

        This implementation returns specification corresponding to unlabeled
        inputs.
        """
        return (self.get_input_space(), self.get_input_source())

    def get_monitoring_channels(self, data):
        X = data
        if X is not None:
            X_hat = self.feed_forward(X, method="MF")
        else:
            X_hat = None
        rval = self.top_rbm.get_monitoring_channels(X_hat)
        X_r = self.reconstruct(X)

        dbn_recons_cost = self.rbms[0].visible_layer.recons_cost(X, X_r)
        rval["dbn_reconstruction_cost"] = dbn_recons_cost
        return rval

    def get_input_space(self):
        return self.input_space

    def reconstruct(self, V):
        X_hat = self.feed_forward(V, method="SAMPLING")
        X_hat_r = self.top_rbm.reconstruct(X_hat)
        V_r = self.generate(self.rbms[-2], X_hat_r)
        return V_r

    def get_weights_topo(self, level=0):
        raise NotImplementedError()

    def get_weights2(self, level=0, method="SAMPLING", niter=100):
        """
        .. todo::

            WRITEME
        """
        assert level < len(self.rbms)
        rbm = self.rbms[level]

        # Hacking a mf inference on the top level (or smapling maybe)
        # We need the identity to set each hidden unit for its batch.
        identity = sharedX(np.identity(rbm.hidden_layers[0].detector_layer_dim))
        state = None, identity.copy()
        layer_above = rbm.hidden_layers[0]
        layer_below = rbm.visible_layer

        for s in xrange(niter):
            state_above = layer_above.downward_state(state)
            if method == "SAMPLING":
                state_below = layer_below.upward_state(layer_below.sample(state_below=None, state_above=state_above,
                                                                                                                layer_above=layer_above,
                                                                                                                theano_rng=self.theano_rng))
                #   Set the diagonal to the offset + 1 if centered, else 1
                state = layer_above.sample(state_below,
                                                         state_above=None, theano_rng=self.theano_rng)
            elif method == "MF":
                state_below = layer_below.upward_state(layer_below.mf_update(state_above,
                                                                                                                layer_above))
                state = layer_above.mf_update(state_below,
                                                         state_above=None)
            if s == niter - 1:
                state = layer_above.mf_update(state_below,
                                                                   state_above=None)
            state = fill_diagonal(state[0], 1), state[1]

        state = state[0]
        if level > 0:
            state = self.generate(rbm, state, to_level=1)
        hidden = self.rbms[0].hidden_layers[0]
        W = hidden.downward_message(state)
        return W.T.eval()

    def get_weights(self, level=0):
        state = sharedX(np.identity(self.rbms[0].visible_layer.nvis))
        if level > 0:
            state = self.feed_forward(state, method="MF", level=level-1)
        W = self.rbms[level].hidden_layers[0].transformer.lmul(state)
        return W.eval()

    def get_weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.rbms[0].hidden_layers[0].get_weights_view_shape()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return self.rbms[0].hidden_layers[0].get_weights_format()


class DBN_Inference_Procedure(InferenceProcedure):

    def mf(self, X, Y = None, return_history=False, niter=None, block_grad=None):
        dbn = self.dbn
        X_hat = dbn.feed_forward(X, method="MF")
        return dbn.top_rbm.inference_procedure.mf(X_hat, Y, return_history, niter, block_grad)

class DBN_Sampling_Procedure(SamplingProcedure):

    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None, num_steps=1):
        dbn = self.dbn
        X = layer_to_state[dbn.visible_layer]
        X_hat = dbn.feed_forward(X, method="SAMPLING")
        layer_to_state[dbn.visible_layer] = X_hat
        return dbn.top_rbm.sampling_procedure.sample(layer_to_state, theano_rng,
                                                                                      layer_to_clamp, num_steps)
