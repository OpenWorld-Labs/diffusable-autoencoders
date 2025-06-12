import torch
from torch import nn
from torchtyping import TensorType

from owl_vaes.data.audio_loader import get_audio_loader
from owl_vaes.nn.audio_blocks import (
    DecoderBlock,
    EncoderBlock,
    WNConv1d,
    get_activation,
)

class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] = [1, 2, 4, 8],
        strides: list[int] = [2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


class AudioAutoEncoder(nn.Module):
    """
    Complete Audio AutoEncoder combining OobleckEncoder and OobleckDecoder.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # Extract parameters from config with defaults
        in_channels = getattr(config, "in_channels", 2)
        out_channels = getattr(config, "out_channels", 2)
        channels = getattr(config, "channels", 128)
        latent_dim = getattr(config, "latent_dim", 32)
        c_mults = getattr(config, "c_mults", [1, 2, 4, 8])
        strides = getattr(config, "strides", [2, 4, 8, 8])
        use_snake = getattr(config, "use_snake", False)
        antialias_activation = getattr(config, "antialias_activation", False)
        use_nearest_upsample = getattr(config, "use_nearest_upsample", False)
        final_tanh = getattr(config, "final_tanh", True)
        self.eq = getattr(config, 'eq', False)

        self.encoder = OobleckEncoder(
            in_channels=in_channels,
            channels=channels,
            latent_dim=latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
        )

        self.decoder = OobleckDecoder(
            out_channels=out_channels,
            channels=channels,
            latent_dim=latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
            use_nearest_upsample=use_nearest_upsample,
            final_tanh=final_tanh,
        )

        # Calculate total stride for shape validation
        self.total_stride = 1

        for stride in strides:
            self.total_stride *= stride

    def encode(self, x: TensorType) -> TensorType:
        return self.encoder(x)

    def decode(self, z: TensorType) -> TensorType:
        return self.decoder(z)

    def forward(self, x: TensorType) -> tuple[TensorType, TensorType]:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input audio tensor of shape (batch, channels, time)

        Returns:
            Tuple: (reconstructed_audio, latent_representation)
        """
        z = self.encode(x)
        x_rec = self.decode(z)

        if self.eq:
            n = z.shape[-1]
            n = n // 3
            z_1 = z[:,:,:2*n] # First 2/3
            z_2 = z[:,:,n:] # Last 2/3
            rec_1 = self.decode(z_1)
            rec_2 = self.decode(z_2)
            return x_rec, z, (rec_1, rec_2)
        else:
            return x_rec, z

    def get_compression_ratio(self) -> int:
        return self.total_stride



if __name__ == "__main__":
    from ..data import get_loader
    from .music2latent_fe import AmplitudeCompressedComplexSTFT as AudioFE

    audio_fe = AudioFE(
        window_fn="hann",
        n_fft=2048,
        sampling_rate=44100,
        alpha=0.5,
        beta=1.0,
        n_hops=4,
        learnable_window=False,
    )

    loader = get_loader("local_cod_audio", 4, root = "../cod_download/raw")
    sample = next(iter(loader))

    #myEncoder = OobleckEncoder(
    #    2, 128, latent_dim = 128, 
    #    use_snake=True, antialias_activation=True,
    #    strides = [3,5,7,7]
    #)
    #myDecoder = OobleckDecoder(
    #    2, 128, antialias_activation=True, use_nearest_upsample=True
    #)

    with torch.no_grad():
        print(sample.shape)
        x = audio_fe.forward(sample)
            
        print(x.shape)
        exit()
        #latent = myEncoder(sample)
        print(latent.shape)

        print("EQ Tests")
        n = sample.shape[-1]
        n_1 = n // 3

        x_1 = sample[:,:,:2*n_1]
        x_2 = sample[:,:,n_1:]

        print(x_1.shape)
        print(x_2.shape)

        z_1 = myEncoder(x_1)
        z_2 = myEncoder(x_2)

        print(z_1.shape)
        print(z_2.shape)



    # 128 -> 172

    exit()
    model_config = {
        "in_channels": 2,
        "out_channels": 2,
        "use_snake": True,
        "antialias_activation": True,
        "use_nearest_upsample": True,
    }

    model = AudioAutoEncoder(model_config)

    latent = myEncoder(sample)
    decoded = myDecoder(latent)

    print("Latent: ", latent)
    print("Latent: ", latent.shape)
    print("Latent: ", latent.dtype)

    print("Decoded: ", decoded)
    print("Decoded: ", decoded.shape)
    print("Decoded: ", decoded.dtype)

    print("MSE:", torch.mean((decoded - sample).pow(2)))

    full_fwd_pass = model(sample)

    print("MSE for full pass:", torch.mean((full_fwd_pass[0] - sample).pow(2)))
