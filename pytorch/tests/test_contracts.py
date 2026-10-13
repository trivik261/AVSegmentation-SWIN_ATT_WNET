import importlib.util
import unittest
import numpy as np
from avseg.tiling import sliding_window, decode
from avseg.preprocess import coordinates

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CV = importlib.util.find_spec("cv2") is not None


class TilingContracts(unittest.TestCase):
    def test_probabilities_preserved_and_every_edge_covered(self):
        calls = []
        def predict(patch):
            calls.append(patch.shape)
            return (np.broadcast_to(np.array([.15,.85])[:,None,None],(2,32,32)),
                    np.broadcast_to(np.array([.1,.7,.2])[:,None,None],(3,32,32)))
        p1,p2=sliding_window(np.zeros((6,47,61)),predict,32,.25)
        self.assertGreater(len(calls),1)
        self.assertEqual(p2.shape,(3,47,61))
        np.testing.assert_allclose(p1[1],.85,atol=1e-6)
        np.testing.assert_allclose(p2[1],.7,atol=1e-6)

    def test_global_coordinates_are_not_reset_for_patches(self):
        image=np.concatenate((np.zeros((4,47,61)),coordinates(47,61)))
        def predict(patch):
            y=patch[-2]; x=patch[-1]
            return np.stack((1-y,y)),np.stack((1-x,x,np.zeros_like(x)))
        p1,p2=sliding_window(image,predict,32,.25)
        np.testing.assert_allclose(p1[1],image[-2],atol=1e-6)
        np.testing.assert_allclose(p2[1],image[-1],atol=1e-6)

    def test_small_image_returns_original_shape(self):
        def predict(patch):
            return np.ones((2,32,32))*.5,np.ones((3,32,32))/3
        p1,p2=sliding_window(np.zeros((6,7,9)),predict,32)
        self.assertEqual(p1.shape,(2,7,9))
        self.assertEqual(p2.shape,(3,7,9))

    def test_logits_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"probabilities"):
            sliding_window(np.zeros((6,32,32)),lambda _: (np.ones((2,32,32)),np.ones((3,32,32))),32)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            sliding_window(np.zeros((6,32,32)),None,32,1.0)

    def test_predicted_vessels_gate_av_without_ground_truth(self):
        vessels=np.array([[[.9,.1]],[[.1,.9]]])
        av=np.array([[[.1,.1]],[[.8,.8]],[[.1,.1]]])
        _,labels=decode(vessels,av)
        np.testing.assert_array_equal(labels,[[0,1]])

    def test_missing_checkpoint_fails_before_optional_imports(self):
        from avseg.predict import main
        with self.assertRaises(SystemExit) as e:
            main(['--checkpoint','does-not-exist.pth','--image','absent.png',
                  '--dataset','custom','--output','unused'])
        self.assertEqual(e.exception.code,2)


@unittest.skipUnless(HAS_CV,"OpenCV not installed")
class PreprocessingContracts(unittest.TestCase):
    def test_six_channels_and_full_image_coordinates(self):
        from avseg.preprocess import v11_input
        x=v11_input(np.zeros((32,48,3),dtype=np.uint8),'custom')
        self.assertEqual(x.shape,(6,32,48))
        np.testing.assert_array_equal(x[-2:],coordinates(32,48))
        self.assertTrue(np.isfinite(x).all())


@unittest.skipUnless(HAS_TORCH,"PyTorch not installed")
class ModelContracts(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)

    def test_cascade_returns_normalized_probabilities(self):
        import torch
        from avseg.attention_wnet import AttentionWNet
        model=AttentionWNet(6,2,3,filters=[8,16,32,64,128]).eval()
        with torch.no_grad():
            outputs=model(torch.randn(1,6,32,32))
        for out,classes in zip(outputs,(2,3)):
            self.assertEqual(tuple(out.shape),(1,classes,32,32))
            torch.testing.assert_close(out.sum(1),torch.ones(1,32,32))

    def test_auxiliary_head_receives_gradients(self):
        import torch
        from avseg.attention_unet_ds import AttentionUNet
        from avseg.supervision import supervised_loss
        model=AttentionUNet(filters=(8,16,32,64,128),dropout=0,deep_supervision=True)
        out,aux=model(torch.randn(2,1,32,32))
        target=torch.zeros((2,32,32),dtype=torch.long)
        criterion=torch.nn.CrossEntropyLoss()
        loss=supervised_loss(out,aux,target,criterion)
        torch.testing.assert_close(loss,criterion(out,target)+.4*criterion(aux,target))
        loss.backward()
        self.assertEqual(tuple(aux.shape),tuple(out.shape))
        self.assertGreater(model.ds_out.weight.grad.abs().sum().item(),0)

    def test_disabling_supervision_returns_no_auxiliary_prediction(self):
        import torch
        from avseg.attention_unet_ds import AttentionUNet
        model=AttentionUNet(filters=(8,16,32,64,128),deep_supervision=False).eval()
        with torch.no_grad():
            _,aux=model(torch.randn(1,1,32,32))
        self.assertIsNone(aux)


if __name__=='__main__':
    unittest.main()
