using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class DownSampleNet : nn.Module<Tensor, Tensor>
{
    private int _inChannels;
    private Convolution _conv1;
    private BatchNorm2d _bn1;
    private Convolution _conv2;
    private ModuleList<ResidualBlock> _resBlocks1;
    private ResidualBlock _downsampleBlock;
    private ModuleList<ResidualBlock> _resBlocks2;
    private AvgPool2d _pooling1;
    private ModuleList<ResidualBlock> _resBlocks3;
    private AvgPool2d _pooling2;

    public DownSampleNet(int inChannels, int outChannels, DeviceType deviceType = DeviceType.CUDA)
        : base("DownSampleNet")
    {
        Device device = new torch.Device(deviceType);

        _inChannels = inChannels / 2;

        _conv1 = nn.Conv2d(inChannels, outChannels / 2, kernel_size: 3, stride: 2, padding: 1, bias: false, device: device);
        _bn1 = nn.BatchNorm2d(outChannels / 2, device: device);
        _resBlocks1 = nn.ModuleList(new ResidualBlock("ResBlock1", outChannels / 2, outChannels / 2, device));
        _conv2 = nn.Conv2d(outChannels / 2, outChannels, kernel_size: 3, stride: 2, padding: 1, bias: false, device: device);
        _downsampleBlock = new ResidualBlock("DownSample", outChannels / 2, outChannels, device, downsample: _conv2, stride: 2);
        _resBlocks2 = nn.ModuleList(new ResidualBlock("ResBlock2", outChannels, outChannels, device));
        _pooling1 = nn.AvgPool2d(kernel_size: 3, stride: 2, padding: 1);
        _resBlocks3 = nn.ModuleList(new ResidualBlock("ResBlock3", outChannels, outChannels, device));
        _pooling2 = nn.AvgPool2d(kernel_size: 3, stride: 2, padding: 1);


        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _conv1.forward(x);
        x = _bn1.forward(x);
        x = F.relu(x);
        foreach (var block in _resBlocks1)
            x = block.forward(x);
        x = _downsampleBlock.forward(x);
        foreach (var block in _resBlocks2)
            x = block.forward(x);
        x = _pooling1.forward(x);
        foreach (var block in _resBlocks3)
            x = block.forward(x);
        x = _pooling2.forward(x);
        return x;
    }
}
