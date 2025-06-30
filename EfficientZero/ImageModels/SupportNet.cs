using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;


namespace EfficientZero.ImageModels;

public class SupportNet : nn.Module<Tensor, Tensor>
{
    private int _flattenSize;
    private BatchNorm2d _bn;
    private Conv2d _conv;
    private MLP _fc;


    public SupportNet(int numBlocks, int numChannels, int reducedChannels, int flattenSize, int fc_layers, int outputSupportSize, bool initZero, DeviceType deviceType = DeviceType.CUDA)
        : base("Support")
    {
        Device device = new torch.Device(deviceType);

        _flattenSize = flattenSize;


        _conv = nn.Conv2d(numChannels, reducedChannels, kernel_size: 3, stride: 1, padding: 1, bias: false, device: device);
        _bn = nn.BatchNorm2d(reducedChannels, device: device);
        _fc = new MLP(flattenSize, fc_layers, outputSupportSize, device, initZero: initZero);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _conv.forward(x);
        x = _bn.forward(x);
        //var xs = x.data<float>().ToArray();
        x = F.relu(x);
        x = x.reshape(-1, _flattenSize);
        x = _fc.forward(x);

        return x;
    }
}
