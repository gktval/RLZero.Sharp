using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class RepresentationNet : nn.Module<Tensor, Tensor>
{
    private int[] _obsShape;
    private BatchNorm2d _bn;
    private Conv2d _conv;
    private ModuleList<ResidualBlock> _resBlocks1;
    private bool _downsample;
    private DownSampleNet _downsampleNet;


    public RepresentationNet(int[] obsShape, int numBlocks, int numChannels, bool downsample, DeviceType deviceType = DeviceType.CUDA)
        : base("Representation")
    {
        Device device = new torch.Device(deviceType);

        _obsShape = obsShape;
        _downsample = downsample;

        if (_downsample)
            _downsampleNet = new DownSampleNet(obsShape[0], numChannels);

        _conv = nn.Conv2d(obsShape[0], numChannels, kernel_size: 3, stride: 1, padding: 1, bias: false, device: device);
        _bn = nn.BatchNorm2d(numChannels, device: device);
        _resBlocks1 = nn.ModuleList(new ResidualBlock("ResBlockRep", numChannels, numChannels, device));

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        if (_downsample)
            x = _downsampleNet.forward(x);
        else
        {
            x = _conv.forward(x);
            //x = _bn.forward(x);
            x = F.relu(x);
        }

        foreach(var block in _resBlocks1)
            x = block.forward(x);

        return x;
    }
}
