using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class DynamicNet : nn.Module
{
    private bool _isContinuous;
    private bool _actionEmbedding;
    private int _actionEmbeddingDim;
    private int _numChannels;
    private int _actionSpaceSize;

    private Conv2d _conv1x1;
    private LayerNorm _ln;
    private Conv2d _conv1;
    private BatchNorm _bn;
    private ModuleList<ResidualBlock> _resBlocks;

    public DynamicNet(int numBlocks, int numChannels, int actionSpaceSize, bool isContinuous = false, bool actionEmbedding = false, int actionEmbeddingDim = 32, DeviceType deviceType = DeviceType.CUDA)
        : base("Dynamic")
    {
        Device device = new torch.Device(deviceType);

        _isContinuous = isContinuous;
        _actionEmbedding = actionEmbedding;
        _actionEmbeddingDim = actionEmbeddingDim;
        _numChannels = numChannels;
        _actionSpaceSize = actionSpaceSize;

        if (actionEmbedding)
        {
            int actionSize = isContinuous ? actionSpaceSize : 1;
            _conv1x1 = nn.Conv2d(actionSpaceSize, actionEmbeddingDim, 1, device: device);
            _ln = nn.LayerNorm(new long[] { actionEmbeddingDim, 6, 6 }, device: device);
            _conv1 = nn.Conv2d(numChannels + actionEmbeddingDim, numChannels, kernel_size: 3, stride: 1, padding: 1, bias: false, device: device);
        }
        else
            _conv1 = nn.Conv2d(numChannels + actionSpaceSize, numChannels, kernel_size: 3, stride: 1, padding: 1, bias: false, device: device);

        _bn = nn.BatchNorm2d(numChannels, device: device);
        _resBlocks = nn.ModuleList(new ResidualBlock("ResBlockDynamic", numChannels, numChannels, device));

        RegisterComponents();
    }

    public Tensor Forward(Tensor latent, Tensor action)
    {
        Tensor actionPlace;
        if (!_isContinuous)
        {
            //var ap = torch.ones(latent.shape[0], 1, latent.shape[2], latent.shape[3], dtype: ScalarType.Float32).cuda();

            //actionPlace = (action.unsqueeze(2).unsqueeze(3) * ap / _actionSpaceSize);

            actionPlace = torch.ones(new long[] { latent.shape[0], 1, latent.shape[2], latent.shape[3] }).cuda().to_type(ScalarType.Float32);
            var ap2 = action[.., .., TensorIndex.None, TensorIndex.None] * actionPlace / _actionSpaceSize;
            actionPlace = ap2;
        }
        else
        {
            long[] latentReshape = latent.shape.Concat<long>(new long[] { 1, 1 }).ToArray();

            actionPlace = latent.reshape(latentReshape).repeat(1, 1, latent.shape[latent.shape.Length - 2], latent.shape[latent.shape.Length - 1]);
        }

        if (_actionEmbedding)
        {
            actionPlace = _conv1x1.forward(actionPlace);
            actionPlace = _ln.forward(actionPlace);
            actionPlace = nn.functional.relu(actionPlace);
        }

        var x = torch.concat(new[] { latent, actionPlace }, 1);
        x = _conv1.forward(x);
        //x = _bn.forward(x);
        foreach (var block in _resBlocks)
            x = block.forward(x);

        //var newLatent = output[.., 0..latentSize];
        //var rewardLogits = output[.., latentSize..];
        return x;
    }


}
