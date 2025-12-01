# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any

import torch
from hypothesis import given
from hypothesis import strategies as st

from generic_neuromotor_interface.networks import (
    DiscreteGesturesArchitecture,
    HandwritingArchitecture,
    HandwritingConformer,
    MaskAug,
    MPFPerChannelFeatureExtractor,
    MultivariatePowerFrequencyFeatures,
    RotationInvariantMPFMLP,
    SetTransformerSpatialEncoder,
    WristArchitecture,
)


class TestRotationInvariantMPFMLP(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=16))
    def test_invalid_adjacent_cov(
        self,
        num_channels: int,
    ) -> None:
        max_adjacent_cov = num_channels // 2

        # test with max allowed adjacent cov
        RotationInvariantMPFMLP(
            num_channels=num_channels,
            num_freqs=1,
            hidden_dims=[1],
            num_adjacent_cov=max_adjacent_cov,
        )

        # test with one above
        with self.assertRaises(ValueError):
            RotationInvariantMPFMLP(
                num_channels=num_channels,
                num_freqs=1,
                hidden_dims=[1],
                num_adjacent_cov=max_adjacent_cov + 1,
            )


class TestRotationInvariantLSTM(unittest.TestCase):
    def test_forward(self) -> None:
        # set some constants
        batch_size = 5
        num_channels = 16
        num_tsteps = 500
        output_dim = 1

        # sample random data
        data = torch.randn(batch_size, num_channels, num_tsteps)

        # assemble network
        network = WristArchitecture(
            num_channels=num_channels,
            hidden_dims=[512],
            lstm_hidden_dim=512,
            lstm_num_layers=2,
            output_dim=output_dim,
        )

        # check that the number of parameters is exactly equal to how many
        # we had when running the scaling plots experiments
        num_params = sum(p.numel() for p in network.parameters())
        self.assertEqual(num_params, 4400129)

        # run network forward pass
        output = network(data)

        # check output shape
        output_num_tsteps = len(
            torch.arange(num_tsteps)[network.left_context :: network.stride]
        )
        self.assertEqual(output.shape, (batch_size, output_dim, output_num_tsteps))


WRIST_MODEL_MPF_PARAMS = {
    "window_length": 200,
    "stride": 40,
    "n_fft": 64,
    "fft_stride": 10,
}

HANDWRITING_MODEL_MPF_PARAMS = {
    "window_length": 160,
    "stride": 40,
    "n_fft": 64,
    "fft_stride": 10,
}


class TestMultivariatePowerFrequencyFeatures(unittest.TestCase):
    @given(
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=16),
        st.sampled_from([WRIST_MODEL_MPF_PARAMS, HANDWRITING_MODEL_MPF_PARAMS]),
    )
    def test_left_context(
        self,
        batch_size: int,
        num_channels: int,
        mpf_parameters: dict[str, Any],
    ) -> None:
        # assemble module
        module = MultivariatePowerFrequencyFeatures(**mpf_parameters)

        # check that module returns length 1 sequence when given
        # input of length left_context + 1
        data = torch.randn(batch_size, num_channels, module.left_context + 1)
        output = module(data)
        self.assertEqual(output.shape[-1], 1)

        # check that module raises RuntimeError when sequence length
        # is left_context
        with self.assertRaises(RuntimeError):
            data = torch.randn(batch_size, num_channels, module.left_context)
            module(data)


class TestDiscreteGesturesArchitecture(unittest.TestCase):
    def test_forward(self) -> None:
        # set some constants
        batch_size = 5
        num_channels = 16
        num_tsteps = 500

        # sample random data
        data = torch.randn(batch_size, num_channels, num_tsteps)

        # assemble network
        network = DiscreteGesturesArchitecture(input_channels=num_channels)

        # check that the number of parameters is exactly equal to how many
        # we had when running the scaling plots experiments
        num_params = sum(p.numel() for p in network.parameters())
        self.assertEqual(num_params, 6482953)

        # run network forward pass
        output = network(data)

        # check output shape
        output_num_tsteps = len(
            torch.arange(num_tsteps)[network.left_context :: network.stride]
        )
        self.assertEqual(output.shape, (batch_size, 9, output_num_tsteps))


class TestSetTransformerSpatialEncoder(unittest.TestCase):
    def test_set_transformer_spatial_encoder_shapes(self) -> None:
        """Test that SetTransformerSpatialEncoder produces correct output shapes."""
        B, T, C, F = 2, 5, 16, 96  # F = num_freqs * num_channels = 6 * 16
        D = 64

        encoder = SetTransformerSpatialEncoder(
            in_feat_dim=F,
            hidden_dim=D,
            num_heads=4,
            num_sab_layers=2,
            ff_dim=256,
        )
        x = torch.randn(B, T, C, F)
        z = encoder(x)
        self.assertEqual(z.shape, (B, T, D))

    def test_set_transformer_with_different_feat_dim(self) -> None:
        """Test SetTransformer with input projection when feat_dim != hidden_dim."""
        B, T, C, F = 2, 5, 16, 96
        D = 128  # Different from F

        encoder = SetTransformerSpatialEncoder(
            in_feat_dim=F,
            hidden_dim=D,
            num_heads=4,
            num_sab_layers=2,
            ff_dim=256,
        )
        x = torch.randn(B, T, C, F)
        z = encoder(x)
        self.assertEqual(z.shape, (B, T, D))


class TestMPFPerChannelFeatureExtractor(unittest.TestCase):
    def test_mpf_per_channel_feature_extractor_shapes(self) -> None:
        """Test that MPFPerChannelFeatureExtractor produces correct output shapes."""
        B, num_freqs, C, T = 2, 6, 16, 10

        extractor = MPFPerChannelFeatureExtractor(
            num_channels=C,
            num_freqs=num_freqs,
        )
        # Input: (B, freq, C, C, T)
        mpf_input = torch.randn(B, num_freqs, C, C, T)
        output = extractor(mpf_input)
        # Output should be (B, T, C, freq * C)
        expected_F = num_freqs * C
        self.assertEqual(output.shape, (B, T, C, expected_F))

    def test_mpf_feature_extractor_assertions(self) -> None:
        """Test that MPFPerChannelFeatureExtractor validates dimensions."""
        B, num_freqs, C, T = 2, 6, 16, 10

        extractor = MPFPerChannelFeatureExtractor(
            num_channels=C,
            num_freqs=num_freqs,
        )
        # Test with correct shape
        mpf_input = torch.randn(B, num_freqs, C, C, T)
        output = extractor(mpf_input)
        self.assertEqual(output.shape, (B, T, C, num_freqs * C))

        # Test with wrong frequency dimension
        wrong_freqs = num_freqs + 1
        mpf_input_wrong = torch.randn(B, wrong_freqs, C, C, T)
        with self.assertRaises(AssertionError):
            extractor(mpf_input_wrong)


class TestHandwritingArchitectureWithSetTransformer(unittest.TestCase):
    def test_handwriting_architecture_with_set_transformer(self) -> None:
        """Test HandwritingArchitecture integration with SetTransformer."""
        B, C, T = 2, 16, 500
        vocab_size = 100
        num_freqs = 6

        # Create featurizer
        featurizer = MultivariatePowerFrequencyFeatures(
            window_length=160,
            stride=40,
            n_fft=64,
            fft_stride=10,
            frequency_bins=[
                (0, 50),
                (30, 100),
                (100, 225),
                (225, 375),
                (375, 700),
                (700, 1000),
            ],
        )

        # Create specaug (dummy - no augmentation for testing)
        specaug = MaskAug(
            max_num_masks=[0, 0],
            max_mask_lengths=[0, 0],
            dims="TF",
            axes_by_coord={"N": [0], "F": [1], "C": [2, 3], "T": [4]},
            mask_value=0.0,
        )

        # Create encoder (simplified for testing)
        encoder = HandwritingConformer(
            in_dim=64,
            out_dim=vocab_size,
            input_dim=64,
            ffn_dim=128,
            kernel_size=8,
            stride=[1],
            num_heads=4,
            attn_window_size=[16],
            num_layers=1,
            dropout=0.0,
            time_reduction_stride=1,
        )

        # Create architecture with SetTransformer
        model = HandwritingArchitecture(
            num_channels=C,
            vocab_size=vocab_size,
            featurizer=featurizer,
            specgram_augment=specaug,
            encoder=encoder,
            use_set_transformer=True,
            num_freqs=num_freqs,
            set_transformer_hidden_dim=64,
            set_transformer_num_heads=4,
            set_transformer_num_layers=2,
            set_transformer_ff_dim=256,
        )

        # Test forward pass
        data = torch.randn(B, C, T)
        emissions, slc = model(data)
        # Check that emissions have correct shape (vocab_size dimension)
        self.assertEqual(emissions.shape[-1], vocab_size)

    def test_handwriting_architecture_backward_compatibility(self) -> None:
        """Test that HandwritingArchitecture maintains backward compatibility."""
        B, C, T = 2, 16, 500
        vocab_size = 100
        num_freqs = 6

        # Create featurizer
        featurizer = MultivariatePowerFrequencyFeatures(
            window_length=160,
            stride=40,
            n_fft=64,
            fft_stride=10,
            frequency_bins=[
                (0, 50),
                (30, 100),
                (100, 225),
                (225, 375),
                (375, 700),
                (700, 1000),
            ],
        )

        # Create specaug
        specaug = MaskAug(
            max_num_masks=[0, 0],
            max_mask_lengths=[0, 0],
            dims="TF",
            axes_by_coord={"N": [0], "F": [1], "C": [2, 3], "T": [4]},
            mask_value=0.0,
        )

        # Create invariance layer (original approach)
        invariance_layer = RotationInvariantMPFMLP(
            num_channels=C,
            num_freqs=num_freqs,
            hidden_dims=[64],
            offsets=[-1, 0, 1],
            num_adjacent_cov=8,
        )

        # Create encoder
        encoder = HandwritingConformer(
            in_dim=64,
            out_dim=vocab_size,
            input_dim=64,
            ffn_dim=128,
            kernel_size=8,
            stride=[1],
            num_heads=4,
            attn_window_size=[16],
            num_layers=1,
            dropout=0.0,
            time_reduction_stride=1,
        )

        # Create architecture without SetTransformer (backward compatible)
        model = HandwritingArchitecture(
            num_channels=C,
            vocab_size=vocab_size,
            featurizer=featurizer,
            specgram_augment=specaug,
            encoder=encoder,
            invariance_layer=invariance_layer,
            use_set_transformer=False,
        )

        # Test forward pass
        data = torch.randn(B, C, T)
        emissions, slc = model(data)
        # Check that emissions have correct shape
        self.assertEqual(emissions.shape[-1], vocab_size)

    def test_handwriting_architecture_errors(self) -> None:
        """Test that HandwritingArchitecture raises appropriate errors."""
        B, C, T = 2, 16, 100
        vocab_size = 100

        # Create minimal components
        featurizer = MultivariatePowerFrequencyFeatures(
            window_length=160,
            stride=40,
            n_fft=64,
            fft_stride=10,
            frequency_bins=[(0, 50), (30, 100)],
        )

        specaug = MaskAug(
            max_num_masks=[0, 0],
            max_mask_lengths=[0, 0],
            dims="TF",
            axes_by_coord={"N": [0], "F": [1], "C": [2, 3], "T": [4]},
            mask_value=0.0,
        )

        encoder = HandwritingConformer(
            in_dim=64,
            out_dim=vocab_size,
            input_dim=64,
            ffn_dim=128,
            kernel_size=8,
            stride=[1],
            num_heads=4,
            attn_window_size=[16],
            num_layers=1,
            dropout=0.0,
            time_reduction_stride=1,
        )

        # Test error when use_set_transformer=True but num_freqs not provided
        with self.assertRaises(ValueError):
            HandwritingArchitecture(
                num_channels=C,
                vocab_size=vocab_size,
                featurizer=featurizer,
                specgram_augment=specaug,
                encoder=encoder,
                use_set_transformer=True,
                num_freqs=None,
            )

        # Test error when use_set_transformer=False but invariance_layer not provided
        with self.assertRaises(ValueError):
            HandwritingArchitecture(
                num_channels=C,
                vocab_size=vocab_size,
                featurizer=featurizer,
                specgram_augment=specaug,
                encoder=encoder,
                use_set_transformer=False,
                invariance_layer=None,
            )
