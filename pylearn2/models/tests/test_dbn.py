from pylearn2.costs.dbm import BaseCD
from pylearn2.costs.dbm import VariationalCD
from pylearn2.costs import dbm as dbm_cost

from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.models.dbn import DBN
from pylearn2.models.dbm.dbm import RBM
from pylearn2.models.tests.test_dbm import make_rbm
from pylearn2.models.tests.test_dbm import Test_CD

from pylearn2.utils import sharedX

import numpy as np

from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

config.exception_verbosity='high'
#config.mode='DebugMode'

class Test_DBN_Basics(object):
    @staticmethod
    def make_basic_dbn(dims = [100, 150, 200],
                       batch_sizes=[5, 23],
                       center=False):
        """
        Makes a basic DBN for testing.
        Tests some properties of the DBM layers, such as centering and bias
        initialization.

        Parameters
        ----------
        dims: list of ints.
        batch_sizes: list of ints.
        center: bool
            Whether to use centering with the lower RBM.
        """

        assert len(batch_sizes) == len(dims) - 1

        rbm1 = make_rbm(dims[0], dims[1], batch_sizes[0], center=center)
        rbm2 = make_rbm(dims[1], dims[2], batch_sizes[1], center=False)
        dbn = DBN(rbm1, rbm2)
        for layer1, layer2 in zip(rbm2.hidden_layers, dbn.hidden_layers):
            assert layer1 == layer2
        assert rbm2.visible_layer == dbn.visible_layer
        lower_hidden = dbn.rbms[0].hidden_layers[0]
        upper_visible = dbn.rbms[1].visible_layer
        assert lower_hidden.detector_layer_dim == upper_visible.nvis
        assert lower_hidden.center == upper_visible.center
        assert np.all(lower_hidden.get_biases() == upper_visible.get_biases())
        if upper_visible.center:
            assert np.all(
                lower_hidden.offset.eval() == upper_visible.offset.eval()),\
                "Offsets do not match: \n%r\n%r" %\
                (lower_hidden.offset.eval(), upper_visible.offset.eval())

        return dbn

    @staticmethod
    def make_deeper_dbn(dims = [100, 150, 200, 250], batch_sizes=[5, 13, 23]):
        """
        Maskes a DBN with more layers.
        """

        dbn1 = Test_DBN_Basics.make_basic_dbn(dims[:-1], batch_sizes[:-1])
        rbm3 = make_rbm(dims[-2], dims[-1], batch_sizes[-1])
        dbn = DBN(dbn1, rbm3)
        return dbn

    def test_basic_dbn(self, dims = [100, 200, 100],
                       batch_sizes=[5, 23],
                       center=False):
        """
        Tests the basic DBN.
        """

        dbn = Test_DBN_Basics.make_basic_dbn(dims, batch_sizes, center=center)
        assert dbn is not None
        return dbn

    def test_basic_dbn_with_centering(self, dims = [10, 20, 30],
                                      batch_sizes=[5, 23]):
        """
        Tests basic DBN with centering on the lower RBM.
        """

        dbn = self.test_basic_dbn(dims, batch_sizes, center=True)
        assert dbn is not None

    def test_deeper_dbn(self, dims = [100, 150, 200, 250],
                        batch_sizes=[5, 13, 23]):
        """
        Tests a DBN with 3 RBMs.
        """

        dbn = Test_DBN_Basics.make_deeper_dbn(dims=dims,
                                              batch_sizes=batch_sizes)
        assert dbn is not None

    def test_shared_variables(self):
        """
        Tests the mlp layers.
        """

        dbn = Test_DBN_Basics.make_deeper_dbn()
        mlp = DBN.create_mlp(dbn.rbms)
        for rbm, mlp_layer in zip(dbn.rbms, mlp.layers):
            hidden_layer = rbm.hidden_layers[0]
            hidden_params = hidden_layer.transformer.get_params()

            # MLP get_params() returns the bias as well.
            mlp_layer_params = mlp_layer.get_params()[:-1]
            assert hidden_layer.b == mlp_layer.b
            assert hidden_params == mlp_layer_params,\
                "Params not the same: hidden:\n%r\nmlp layer:\n%r"\
                % (hidden_params, mlp_layer_params)
            for param1, param2 in zip(hidden_params, mlp_layer_params):
                assert np.array_equal(param1.eval(), param2.eval())

    def test_feed_forward(self):
        """
        Tests the feed forward function.
        """

        theano_rng = MRG_RandomStreams(2014+10+8)
        rng = np.random.RandomState([2014,10,7])

        dbn = Test_DBN_Basics.make_deeper_dbn(batch_sizes=[5, 13, 5000])
        X = sharedX(rng.randn(dbn.rbms[0].batch_size,
                              dbn.rbms[0].visible_layer.nvis))
        Y_mf = dbn.feed_forward(X, method="MF")[-1]
        Y_s = dbn.feed_forward(X, method="SAMPLING", theano_rng=theano_rng)[-1]
        assert np.allclose(Y_mf.eval().mean(axis=0),
                           Y_s.eval().mean(axis=0),rtol=1e-1,atol=1e-1)

    def test_generate(self, batch_size=1):
        """
        Tests generation.
        """

        rng = np.random.RandomState([2014,10,19])
        dbn = Test_DBN_Basics.make_deeper_dbn()
        Y = sharedX(rng.randn(batch_size,
                              dbn.top_rbm.hidden_layer.detector_layer_dim))
        X_act = dbn.generate(dbn.top_rbm, Y)

        for rbm in dbn.rbms[::-1]:
            msg = rbm.hidden_layers[0].downward_message(Y)
            z = msg + rbm.visible_layer.bias
            Y = T.nnet.sigmoid(z)
        X_exp = Y
        assert np.all(X_act.eval() == X_exp.eval())


class Test_DBN_as_RBM(object):
    """
    Tests the learning algorithms on DBN on the top RBM.
    """
    def test_CD(self):
        """
        Tests contrastive divergence.
        """

        theano_rng = MRG_RandomStreams(2014+10+9)
        rng = np.random.RandomState([2014,10,7])

        dbn = Test_DBN_Basics.make_deeper_dbn()
        cost = VariationalCD()

        X = sharedX(rng.randn(dbn.rbms[0].batch_size,
                              dbn.rbms[0].visible_layer.nvis))
        X_hat = dbn.feed_forward(X, method="SAMPLING", theano_rng=theano_rng)

        # Get the gradients from RBM as RBM
        pos_grads, up_pos = Test_CD.check_rbm_pos_phase(dbn.top_rbm,
                                                        cost,
                                                        X_hat,
                                                        tol=0.59)
        neg_grads, up_neg = Test_CD.check_rbm_neg_phase(dbn.top_rbm,
                                                        cost,
                                                        X_hat,
                                                        theano_rng=theano_rng,
                                                        tol=0.6)

        # Check the positive gradients.
        for pos in pos_grads:
            assert np.all(
                pos_grads[pos].shape.eval() == neg_grads[pos].shape.eval()),\
                "Gradient shapes do not match for %r (%r vs %r)"\
                % (pos, pos_grads[pos].shape.eval(),
                   neg_grads[pos].shape.eval())

        pos_grads, up_pos = cost._get_positive_phase(dbn, X)
        for grad in pos_grads:
            pos_grads[grad].eval()
            print "pos", grad, pos_grads[grad].shape.eval()
        layer_to_chains = dbn.initialize_chains(X, None, theano_rng)
        for layer in layer_to_chains:
            if isinstance(layer_to_chains[layer], tuple):
                layer_to_chains[layer][0].eval()
                layer_to_chains[layer][1].eval()
            else:
                layer_to_chains[layer].eval()

        # Get sampling updates from DBN as DBN.
        updates, layer_to_chains = dbn.get_sampling_updates(
            layer_to_chains,
            theano_rng,
            num_steps=1,
            return_layer_to_updated=True)

        for update in updates:
            updates[update].eval()

        for layer in layer_to_chains:
            if isinstance(layer_to_chains[layer], tuple):
                layer_to_chains[layer][0].eval()
                layer_to_chains[layer][1].eval()
            else:
                layer_to_chains[layer].eval()

        neg_phase_grads = dbm_cost.negative_phase(dbn, layer_to_chains,
                                                  method=cost.negative_method)
        for grad in neg_phase_grads:
            neg_phase_grads[grad].eval()
            assert np.all(
                pos_grads[grad].shape.eval() == neg_grads[grad].shape.eval()),\
                "%r grad shapes do not match (%r vs %r)"\
                %(grad, pos_grads[grad].shape.eval(),
                  neg_grads[grad].shape.eval())

        neg_grads, up_neg = cost._get_negative_phase(dbn, X)
        for grad in neg_grads:
            neg_grads[grad].eval()

        grads, updates = cost.get_gradients(dbn, X)
        for grad in grads:
            grads[grad].eval()

        for update in updates:
            updates[update].eval()


"""
def test_rbm_dims(dims = [100, 200, 100], batch_sizes = [5, 23, 37]):
    #Function to test that dimensions line up with stacked RBMs.
    assert len(dims) == len(batch_sizes)
"""