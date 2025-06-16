using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.StateModels;

public class DynamicNet : nn.Module
{
    private int latentSize;
    private int actionSize;
    private int fullWidth;
    private LayerNorm _actLn;
    private Linear _actLinear;
    private LayerNorm _dynLn;
    private Linear _dynLinear1;
    private Linear _dynLinear2;
    //private Linear _fc3_2;

    public DynamicNet(int actionSize, int latentSize, int actShape = 32, int dynShape = 128, int supportWidth = 51, DeviceType deviceType = DeviceType.CUDA)
        : base("Dynamic")
    {
        Device device = new torch.Device(deviceType);

        this.latentSize = latentSize;
        this.actionSize = actionSize;
        this.fullWidth = (2 * supportWidth) + 1;


        _actLn = nn.LayerNorm(actShape, device: device);
        _actLinear = nn.Linear(actionSize, actShape, device: device);
        _dynLn = nn.LayerNorm(latentSize + actShape, device: device);
        _dynLinear1 = nn.Linear(latentSize + actShape, dynShape, device: device);
        _dynLinear2 = nn.Linear(dynShape, latentSize + this.fullWidth, device: device);

        RegisterComponents();
    }

    public (Tensor newLatent, Tensor rewardLogits) Forward(Tensor latent, Tensor action)
    {
        if (latent.dim() != 2 || action.dim() != 2)
            throw new ArgumentException("Latent and action tensors must have 2 dimensions.");
        if (latent.shape[1] != latentSize || action.shape[1] != actionSize)
            throw new ArgumentException("Latent and action tensors must have the correct shapes.");

        action = action.to_type(ScalarType.Float32);
        var act_emb = _actLinear.forward(action);
        act_emb = _actLn.forward(act_emb);
        act_emb = F.relu(act_emb);

        var x = torch.concat(new[] { act_emb, latent }, 1);
        x = _dynLn.forward(x);
        x = _dynLinear1.forward(x);
        x = F.relu(x);
        var output = _dynLinear2.forward(x);
        //var out2 = _fc3_2.forward(x);
        var newLatent = output[.., 0..latentSize];
        var rewardLogits = output[.., latentSize..];
        return (newLatent, rewardLogits);
    }
}
