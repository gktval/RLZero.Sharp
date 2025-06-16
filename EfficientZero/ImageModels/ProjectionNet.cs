using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class ProjectionNet : nn.Module<Tensor, Tensor>
{
    private int _inDim;
    private Sequential _layer;
    private Linear _ln1;
    private Linear _ln2;
    private Linear _ln3;

    public ProjectionNet(int inputDim, int hiddenDim, int outDim, DeviceType deviceType = DeviceType.CUDA)
        : base("Projection")
    {
        Device device = new torch.Device(deviceType);

        _inDim = inputDim;

        _ln1 = nn.Linear(inputDim, hiddenDim, device: device);
        _ln2 = nn.Linear(hiddenDim, hiddenDim, device: device);
        _ln3 = nn.Linear(hiddenDim, hiddenDim, device: device);

        _layer = nn.Sequential(
            ("ln1",nn.Linear(inputDim, hiddenDim, device:device)),
            //("bn1", nn.BatchNorm1d(hiddenDim, device:device)),
            ("relu1", nn.ReLU()),

            ("ln2", nn.Linear(hiddenDim, hiddenDim, device:device)),
            //("bn2", nn.BatchNorm1d(hiddenDim,device:device)),
            ("relu2", nn.ReLU()),

            ("ln3", nn.Linear(hiddenDim, outDim, device:device))
            //("bn3", nn.BatchNorm1d(outDim, device:device))
       );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input.reshape(-1, _inDim);

        x =_ln1.forward(x);
        x = F.relu(x);
        x = _ln2.forward(x);
        x = F.relu(x);
        x = _ln3.forward(x);

        //x = _layer.forward(x);
        //input = input.flatten().unsqueeze(0);

        return x;
    }
}
