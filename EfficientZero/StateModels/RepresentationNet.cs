using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.StateModels;

public class RepresentationNet : nn.Module<Tensor, Tensor>
{
    private int _obsSize;
    private LayerNorm _ln;
    private Linear _fc1;
    private Linear _fc2;


    public RepresentationNet(int obsSize, int latentSize, DeviceType deviceType = DeviceType.CUDA)
        : base("Representation")
    {
        Device device = new torch.Device(deviceType);

        _obsSize = obsSize;
        _ln = nn.LayerNorm(obsSize, device:device);
        _fc1 = nn.Linear(obsSize, latentSize, device: device);
        _fc2 = nn.Linear(latentSize, latentSize, device: device);

        RegisterComponents();
    }

    public override Tensor forward(Tensor state)
    {
        if (state.dim() != 2)
        {
            throw new ArgumentException("State must be a 2D tensor.");
        }
        if (state.shape[1] != _obsSize)
        {
            throw new ArgumentException("State shape does not match observation size.");
        }

        state = state.to_type(ScalarType.Float32);
        Tensor output = _ln.forward(state);
        output = _fc1.forward(state);
        output = F.relu(output);
        output = _fc2.forward(output);
        return output;
    }
}
