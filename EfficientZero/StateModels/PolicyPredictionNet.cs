using static TorchSharp.torch;
using TorchSharp.Modules;
using F = TorchSharp.torch.nn.functional;
using TorchSharp;

namespace EfficientZero.Models;

public class PolicyPredictionNet : nn.Module
{
    private int actionSize;
    private int latentSize;
    private int fullWidth;
    private LayerNorm _ln;
    private Linear _fc1;
    private Linear _fc2;
    private Linear _fc3_1;
    //private Linear _fc3_2;

    public PolicyPredictionNet(int actionSize, int latentSize, int fc_units = 64, int supportWidth = 51, DeviceType deviceType = DeviceType.CUDA)
        : base("PolicyPrediction")
    {
        Device device = new torch.Device(deviceType);

        this.actionSize = actionSize;
        this.latentSize = latentSize;
        this.fullWidth = (supportWidth * 2) + 1;
        _ln = nn.LayerNorm(latentSize, device: device);
        _fc1 = nn.Linear(latentSize, fc_units, device: device);
        _fc2 = nn.Linear(fc_units, fc_units, device: device);
        _fc3_1 = nn.Linear(fc_units, actionSize + this.fullWidth, device: device);
        //_fc3_2 = nn.Linear(fc_units, actionSize + this.fullWidth, device: device);

        RegisterComponents();
    }

    public (Tensor policy_logits, Tensor value_logits) Forward(Tensor latent)
    {
        if (latent.dim() != 2)
        {
            throw new ArgumentException("Latent tensor must be 2-dimensional.");
        }
        if (latent.shape[1] != this.latentSize)
        {
            throw new ArgumentException("Latent tensor size does not match latentSize.");
        }

        Tensor x = _ln.forward(latent);
        x = _fc1.forward(latent);
        x = F.relu(x);
        x = _fc2.forward(x);
        x = F.relu(x);
        var out1 = _fc3_1.forward(x);
        //var out2 = _fc3_2.forward(x);
        var valueLogits = out1[.., 0..this.actionSize];
        var policyLogits = out1[.., this.actionSize..];
        return (policyLogits, valueLogits);
    }
}
