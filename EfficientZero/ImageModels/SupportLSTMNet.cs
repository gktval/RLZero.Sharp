using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;
using System;

namespace EfficientZero.ImageModels;

public class SupportLSTMNet : nn.Module
{
    private int _flattenSize;
    private BatchNorm2d _bn;
    private Conv2d _conv;
    private LSTM _lstm;
    private BatchNorm1d _bnRewardSum;
    private MLP _fc;


    public SupportLSTMNet(int numBlocks, int numChannels, int reducedChannels, int flattenSize, int fc_layers, int outputSupportSize, int lstmHiddenSize, bool initZero, DeviceType deviceType = DeviceType.CUDA)
        : base("SupportLSTM")
    {
        Device device = new torch.Device(deviceType);

        _flattenSize = flattenSize;

        _conv = nn.Conv2d(numChannels, reducedChannels, kernel_size: 3, stride: 1, padding: 1, bias: false, device: device);
        _bn = nn.BatchNorm2d(reducedChannels, device: device);
        _lstm = nn.LSTM(inputSize: flattenSize, hiddenSize: lstmHiddenSize, device: device);
        _bnRewardSum = nn.BatchNorm1d(lstmHiddenSize, device: device);
        _fc = new MLP(lstmHiddenSize, fc_layers, outputSupportSize, device, initZero: initZero);

        RegisterComponents();
    }

    public (Tensor reward, (Tensor hiddenC, Tensor hiddenH)) Forward(Tensor x, (Tensor, Tensor) hidden)
    {
        x = _conv.forward(x);
        x = _bn.forward(x);
        x = F.relu(x);
        x = x.reshape(-1, _flattenSize);

        x = x.unsqueeze(0);
        var (lstmX, hiddenOut, other) = _lstm.forward(x, hidden);
        //var xs = x.data<float>().ToArray();

        x = lstmX.squeeze(0);
        //x = F.normalize(x, p: 2, dim: 1);
        x = _bnRewardSum.forward(x);
        //System.Diagnostics.Debug.WriteLine("bn", torch.isfinite(x).all());

        //var xs = x.data<float>().ToArray();
        //if (xs.Any(f => float.IsNaN(f)))
        //    System.Diagnostics.Debug.WriteLine("Has NaN!");
        x = F.relu(x);
        x = _fc.forward(x);

        return (x, (hiddenOut.squeeze(0), other.squeeze(0)));
    }
}
