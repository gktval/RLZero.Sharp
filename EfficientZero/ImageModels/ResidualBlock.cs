using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class ResidualBlock : nn.Module<Tensor, Tensor>
{
    private Convolution _conv1;
    private BatchNorm2d _bn1;
    private Convolution _conv2;
    private BatchNorm2d _bn2;
    private Convolution? _downsample;

    public ResidualBlock(string name, int inChannels, int outChannels,  Device device, Convolution? downsample = null, int stride = 1)
        : base(name)
    {

        _conv1 = Conv3x3(inChannels, outChannels, device, stride);
        _bn1 = nn.BatchNorm2d(outChannels, device: device);
        _conv2 = Conv3x3(outChannels, outChannels, device);
        _bn2 = nn.BatchNorm2d(outChannels, device: device);
        _downsample = downsample;
        
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        Tensor identity = x;

        var output = _conv1.forward(x);
        //output = _bn1.forward(output);
        var xS = output.flatten().data<float>().ToArray();
        output = F.relu(output);

        output = _conv2.forward(output);
        //output = _bn2.forward(output);

        if (_downsample != null)
            identity = _downsample.forward(x);

        output += identity;
        output = F.relu(output);
        return output;
    }

    public Conv2d Conv3x3(int inChannels, int outChannels, Device device, int stride = 1)
    {
        return nn.Conv2d(inChannels, outChannels, kernel_size: 3, stride: stride, padding: 1, bias: false, device: device);
    }
}


