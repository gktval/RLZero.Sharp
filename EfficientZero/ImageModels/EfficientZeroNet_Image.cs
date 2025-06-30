using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;
using EfficientZero.Models;

namespace EfficientZero.ImageModels;

public class EfficientZeroNet_Image : nn.Module, IEfficientZeroNet
{
    private Config _config;
    private int _actionSize;
    private int[] _obs_shape;
    private int _latentSize;
    private int _fc_units; //this is the hidden shape in the model
    private bool _downSample;
    private bool _initZero;
    private int _numBlocks;
    private int _numChannels;
    private int _reducedChannels;
    private bool _actionEmbedding;
    private int _actionEmbeddingSize;
    private int _lstmHiddenSize;

    public Optimizer Optimizer { get; set; }

    private PolicyPredictionNet _policyPredictionNetwork;
    private SupportLSTMNet _rewardPredictionNetwork;
    private DynamicNet _dynamicsNetwork;
    private RepresentationNet _representationNetwork;
    private ProjectionNet _projectionNetwork;
    private ProjectionHeadNet _headnet;

    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> PolicyLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> RewardLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> ValueLoss { get; set; }

    public override IEnumerable<Parameter> parameters(bool recurse = true)
    {
        return base.parameters(recurse);
    }

    public EfficientZeroNet_Image(int actionSize, Config config, DeviceType deviceType = DeviceType.CUDA)
        : base("EfficientZero")
    {
        _config = config;
        _actionSize = actionSize;
        _obs_shape = config.obs_shape;
        _latentSize = config.latent_size;
        _fc_units = config.fc_units;
        _downSample = config.downsample;
        _initZero = config.init_zero;
        _numBlocks = config.num_blocks;
        _numChannels = config.num_channels;
        _reducedChannels = config.reduced_channels;
        _actionEmbedding = config.action_embedding;
        _actionEmbeddingSize = config.action_embedding_dim;
        _lstmHiddenSize = config.lstm_hidden_size;


        bool isContinuous = false;
        int v_num = 1;
        bool value_prefix = true;

        int[] stateShape;
        if (_downSample)
            stateShape = new int[] { _numChannels, (int)Math.Ceiling(_obs_shape[1] / 16f), (int)Math.Ceiling(_obs_shape[2] / 16f) };
        else
            stateShape = new int[] { _numChannels, _obs_shape[1], _obs_shape[2] };

        int stateDim = stateShape[0] * stateShape[1] * stateShape[2];
        int flattenSize = _reducedChannels * stateShape[1] * stateShape[2];

        int _obsShape = _obs_shape[0] * _obs_shape[1] * _obs_shape[2];

        _representationNetwork = new RepresentationNet(_obs_shape, _numBlocks, _numChannels, _downSample, deviceType);

        _dynamicsNetwork = new DynamicNet(_numBlocks, _numChannels, actionSize, actionEmbedding: _actionEmbedding,
            actionEmbeddingDim: _actionEmbeddingSize, deviceType: deviceType);

        _policyPredictionNetwork = new PolicyPredictionNet(_numBlocks, _numChannels, _reducedChannels, flattenSize,
            _fc_units, 51, _actionSize, _initZero, isContinuous, v_num, deviceType);

        if (value_prefix)
            _rewardPredictionNetwork = new SupportLSTMNet(0, _numChannels, _reducedChannels, flattenSize, _fc_units, 51, _lstmHiddenSize, _initZero, deviceType);

        var projectionLayers = config.projection_layers;
        var headLayers = config.prjection_head_layers;
        _projectionNetwork = new ProjectionNet(stateDim, projectionLayers[0], projectionLayers[1], deviceType);
        _headnet = new ProjectionHeadNet(projectionLayers[1], headLayers[0], headLayers[1], deviceType);

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
        x1 = F.normalize(x1, p: 2.0, dim: -1, eps: 1e-5);
        x2 = F.normalize(x2, p: 2.0, dim: -1, eps: 1e-5);

        //var x1d = x1.data<float>().ToArray();
        //var x2d = x2.data<float>().ToArray();

        return -(x1 * x2).sum(dim: 1);
    }

    public void InitParameters()
    {
        var paramList = new List<Parameter>();
        paramList.AddRange(_policyPredictionNetwork.parameters());
        paramList.AddRange(_rewardPredictionNetwork.parameters());
        paramList.AddRange(_dynamicsNetwork.parameters());
        paramList.AddRange(_representationNetwork.parameters());
        paramList.AddRange(_projectionNetwork.parameters());
        paramList.AddRange(_headnet.parameters());

        base.register_parameter("policy", paramList[0]);
        base.register_parameter("reward", paramList[1]);
        base.register_parameter("dynamics", paramList[2]);
        base.register_parameter("representation", paramList[3]);
        base.register_parameter("projection", paramList[4]);
        base.register_parameter("projectionHead", paramList[5]);
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

        //var mu = policy[.., ..288].detach().cpu().flatten().data<float>();
        //var signma = policy[.., 288].detach().cpu().flatten().data<float>();

        return (latentState, value, policy);
    }

    public (Tensor nextLatentState, Tensor valuePrefix, Tensor value, Tensor policy, (Tensor,Tensor) rewardHidden)
        RecurrentInference(Tensor latentState, Tensor action, (Tensor, Tensor) rewardHidden)
    {
        Tensor nextLatentState = Dynamics(latentState, action);
        //var test = nextLatentState.data<float>().ToArray();

        (Tensor valuePrefix, (Tensor,Tensor) outRewardHidden) = RewardPrediction(nextLatentState, rewardHidden);
        (Tensor value, Tensor policy) = PolicyPredict(nextLatentState);

        //value = ScalarConverter.InverseScalarTransform(value);
        //value = value.clip(0, .00005);

        //valuePrefix = ScalarConverter.InverseScalarTransform(valuePrefix);

        return (nextLatentState, valuePrefix, value, policy, outRewardHidden);
    }
    public Tensor Represent(Tensor observation)
    {
        return this._representationNetwork.forward(observation);
    }

    private Tensor Dynamics(Tensor latent, Tensor action)
    {
        return this._dynamicsNetwork.Forward(latent, action);
    }

    private (Tensor value, (Tensor hiddenC, Tensor hiddenH)) RewardPrediction(Tensor latent, (Tensor, Tensor) hidden)
    {
        return this._rewardPredictionNetwork.Forward(latent, hidden);
    }
    private (Tensor policy, Tensor value) PolicyPredict(Tensor latent)
    {
        return this._policyPredictionNetwork.Forward(latent);
    }


    public Tensor Projection(Tensor latent, bool withGrad = true)
    {
        // only the branch of proj + pred can share the gradients
        Tensor proj = _projectionNetwork.forward(latent);

        if (withGrad)
            proj = _headnet.forward(proj);

        return proj;
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
