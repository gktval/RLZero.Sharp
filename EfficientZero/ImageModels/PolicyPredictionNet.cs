using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class PolicyPredictionNet : nn.Module
{
    private bool _isContinuous;
    private int _numChannels;
    private int _flattenSize;
    private float _initStd;
    private float _minStd;
    private int _v_num;

    private ModuleList<Conv2d> _conv1x1_values;
    private Conv2d _conv1x1_policy;
    private ModuleList<BatchNorm2d> _bn_values;
    private BatchNorm2d _bn_policy;

    private ModuleList<ResidualBlock> _resBlocks;
    private ModuleList<MLP> _fc_values;
    private MLP _fc_policy;

    public PolicyPredictionNet(int numBlocks, int numChannels, int reducedChannels, int flattenSize, int fc_layers,
        int valueOutputSize, int policyOutputSize, bool initZero, bool isContinuous = false, int v_num = 1, DeviceType deviceType = DeviceType.CUDA)
        : base("Dynamic")
    {
        Device device = new torch.Device(deviceType);

        _isContinuous = isContinuous;
        _numChannels = numChannels;
        _initStd = 1.0f;
        _minStd = 0.1f;
        _flattenSize = flattenSize;
        _v_num = v_num;

        _resBlocks = nn.ModuleList(new ResidualBlock("ResBlockPP", numChannels, numChannels, device));

        List<Conv2d> valueConvs = new List<Conv2d>();
        for (int v = 0; v < v_num; v++)
            valueConvs.Add(nn.Conv2d(numChannels, reducedChannels, 1, device: device));
        _conv1x1_values = nn.ModuleList<Conv2d>(valueConvs.ToArray());
        _conv1x1_policy = nn.Conv2d(numChannels, reducedChannels, 1, device: device);

        List<BatchNorm2d> valueBns = new List<BatchNorm2d>();
        for (int v = 0; v < v_num; v++)
            valueBns.Add(nn.BatchNorm2d(reducedChannels, device: device));
        _bn_values = nn.ModuleList<BatchNorm2d>(valueBns.ToArray());
        _bn_policy = nn.BatchNorm2d(reducedChannels, device: device);

        List<MLP> valueMLPs = new List<MLP>();
        bool doInitZero = isContinuous ? false : initZero;
        for (int v = 0; v < v_num; v++)
            valueMLPs.Add(new MLP(flattenSize, fc_layers, valueOutputSize, device, initZero: doInitZero));
        _fc_values = nn.ModuleList<MLP>(valueMLPs.ToArray());
        int policySize = _isContinuous ? 64 : policyOutputSize;
        _fc_policy = new MLP(flattenSize, fc_layers, policySize, device, initZero: initZero);

        RegisterComponents();
    }

    public (Tensor values, Tensor policy) Forward(Tensor x)
    {
        foreach(var block in _resBlocks)
            x = block.forward(x);

        List<Tensor> values = new List<Tensor>();
        for (int v = 0; v < _v_num; v++)
        {
            var val = _conv1x1_values[v].forward(x);
            val = _bn_values[v].forward(val);
            val = F.relu(val);
            val = val.reshape(-1, _flattenSize);
            //var xsV = val.data<float>().ToArray();
            val = _fc_values[v].forward(val);
            values.Add(val);
        }  
        
        var policy = _conv1x1_policy.forward(x);
        policy = _bn_policy.forward(policy);
        policy = F.relu(policy);
        policy = policy.reshape(-1, _flattenSize);
        //var xsP = policy.data<float>().ToArray();
        policy = _fc_policy.forward(policy);

        if(_isContinuous)
        {
            int actionSpaceSize = (int)(policy.shape[-1] / 2);
            policy[.., ..actionSpaceSize] = 5 * torch.tanh(policy[.., ..actionSpaceSize] / 5); //soft clamp mu
            policy[.., actionSpaceSize..] = (F.softplus(policy[.., actionSpaceSize..] + _initStd) + _minStd); //.clip(0, 5)  # same as Dreamer-v3
        }

        return (torch.stack(values).squeeze(0), policy);

    }


}
