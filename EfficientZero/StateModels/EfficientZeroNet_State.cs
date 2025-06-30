using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;
using EfficientZero.Models;

namespace EfficientZero.StateModels;

public class EfficientZeroNet_State : nn.Module, IEfficientZeroNet
{
    private Config _config;
    private int _actionSize;
    private int _observationSize;
    private int _latentSize;
    private int _act_units; //this is the hidden shape in the model
    private int _dyn_units; //this is the hidden shape in the model
    private int _fc_units; //this is the hidden shape in the model
    private int _supportWidth;

    public Optimizer Optimizer { get; set; }

    private PolicyPredictionNet _policyPredictionNetwork;
    private RewardPredictionNet _rewardPredictionNetwork;
    private DynamicNet _dynamicsNetwork;
    private RepresentationNet _representationNetwork;

    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> PolicyLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> RewardLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> ValueLoss { get; set; }

    public override IEnumerable<Parameter> parameters(bool recurse = true)
    {
        return base.parameters(recurse);
    }
    //public List<Parameter> Parameters { get; set; }

    public EfficientZeroNet_State(int actionSize, int observationSize, int rewardHiddenSize, Config config, DeviceType deviceType = DeviceType.CUDA)
        : base("EfficientZero")
    {
        _config = config;
        _actionSize = actionSize;
        _observationSize = observationSize;
        _latentSize = config.latent_size;
        _fc_units = config.fc_units;
        _act_units = config.act_units;
        _dyn_units = config.dyn_units;
        _supportWidth = config.support_width;

        _policyPredictionNetwork = new PolicyPredictionNet(_actionSize, _latentSize, _fc_units, _supportWidth, deviceType);
        _rewardPredictionNetwork = new RewardPredictionNet(_latentSize, rewardHiddenSize, _fc_units, _supportWidth, deviceType);

        _dynamicsNetwork = new DynamicNet(this._actionSize, _latentSize, _act_units, _dyn_units, _supportWidth, deviceType);
        _representationNetwork = new RepresentationNet(_observationSize, _latentSize, deviceType);

        PolicyLoss = nn.CrossEntropyLoss();
        RewardLoss = nn.CrossEntropyLoss();
        ValueLoss = nn.CrossEntropyLoss();

        RegisterComponents();

        InitParameters();
    }

    public void Eval()
    {
        eval();
    }

    public void Train()
    {
        train();
    }

    public void Save(string location)
    {
        base.save(location);
    }

    public void Load(string location, bool strict = true)
    {
        base.load(location, strict: true);
    }


    public Tensor ConsistencyLoss(Tensor x1, Tensor x2)
    {
        x1 = F.normalize(x1, p: 2.0, dim: 0, eps: 1e-5);
        x2 = F.normalize(x2, p: 2.0, dim: 0, eps: 1e-5);
        return -(x1 * x2).sum(dim: 0);
    }

    public void InitParameters()
    {
        var paramList = new List<Parameter>();
        paramList.AddRange(_policyPredictionNetwork.parameters());
        paramList.AddRange(_rewardPredictionNetwork.parameters());
        paramList.AddRange(_dynamicsNetwork.parameters());
        paramList.AddRange(_representationNetwork.parameters());

        base.register_parameter("policy", paramList[0]);
        base.register_parameter("reward", paramList[1]);
        base.register_parameter("dynamics", paramList[2]);
        base.register_parameter("representation", paramList[3]);
    }
    public void InitOptimizer(double learningRate)
    {
        this.Optimizer = new SGD(parameters(), learningRate, weight_decay: _config.weight_decay, momentum: _config.momentum);
    }

    public (Tensor state, Tensor value, Tensor policy) InitialInference(Tensor observation)
    {
        var latentState = Represent(observation);
        (Tensor value, Tensor policy) = PolicyPredict(latentState);

        //if (training)
        //    return (latentState, value, policy);

        //var scalarValue = ScalarConverter.InverseScalarTransform(value);
        //scalarValue = (float)Math.Clamp(scalarValue, 0, .00005);

        return (latentState, value, policy);
    }

    public (Tensor nextLatentState, Tensor valuePrefix, Tensor value, Tensor policy, (Tensor,Tensor) rewardHidden) 
        RecurrentInference(Tensor latentState, Tensor action, (Tensor, Tensor) rewardHidden)
    {
        (Tensor nextLatentState, Tensor reward) = Dynamics(latentState, action);
        (Tensor valuePrefix, Tensor outRewardHidden) = RewardPrediction(nextLatentState);
        (Tensor value, Tensor policy) = PolicyPredict(nextLatentState);

        //value = ScalarConverter.InverseScalarTransform(value);
        //value = value.clip(0, .00005);

        //valuePrefix = ScalarConverter.InverseScalarTransform(valuePrefix);

        return (nextLatentState, valuePrefix, value, policy, (reward,reward));
    }

    private (Tensor policy, Tensor value) PolicyPredict(Tensor latent)
    {
        return this._policyPredictionNetwork.Forward(latent);
    }

    private (Tensor value, Tensor policy) RewardPrediction(Tensor latent)
    {
        return this._rewardPredictionNetwork.Forward(latent);
    }

    private (Tensor latent, Tensor reward) Dynamics(Tensor latent, Tensor action)
    {
        return this._dynamicsNetwork.Forward(latent, action);
    }

    public Tensor Represent(Tensor observation)
    {
        return this._representationNetwork.forward(observation);
    }

    public Tensor Projection(Tensor latent, bool withGrad = true)
    {
        //do nothing
        return latent;
    }

    //public void Save(string logDir)
    //{
    //    string predFilename = Path.Combine(logDir, "pred.pt");
    //    _policyPredictionNetwork.save(predFilename);

    //    string dynaFilename = Path.Combine(logDir, "dyna.pt");
    //    _dynamicsNetwork.save(predFilename);

    //    string repFilename = Path.Combine(logDir, "rep.pt");
    //    _representationNetwork.save(predFilename);
    //}
}
