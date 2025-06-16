using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class ProjectionHeadNet : nn.Module<Tensor, Tensor>
{
    private Sequential _layer;
    private Linear _ln1;
    private Linear _ln2; 

    public ProjectionHeadNet(int inputDim, int hiddenDim, int outDim, DeviceType deviceType = DeviceType.CUDA)
        : base("Projection")
    {
        Device device = new torch.Device(deviceType);

        _ln1 = nn.Linear(inputDim, hiddenDim, device: device);
        _ln2 = nn.Linear(hiddenDim, outDim, device: device);

        _layer = nn.Sequential(new List<nn.Module<Tensor, Tensor>>()
        {
            nn.Linear(inputDim, hiddenDim, device:device),
            //nn.BatchNorm1d(hiddenDim, device:device),
            nn.ReLU(),
            nn.Linear(hiddenDim, outDim, device:device),
        });

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _ln1.forward(input);
        x = F.relu(x);
        x = _ln2.forward(x);
        return x;
        //return _layer.forward(input);
    }
}
