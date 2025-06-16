using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;

namespace EfficientZero.Models;

public class CartEfficientNet_Batch : nn.Module
{
    private Dictionary<string, dynamic> config;
    private int actionSize;
    private int obsSize;
    private int latentSize;
    private int supportWidth;
    private int lstmHiddenSize;
    private CartPredictionNet predictionNetwork;
    private CartDynamicLSTM dynamicsNetwork;
    private CartDynamicNet dynamicsNetworkAlternative;
    private CartRepresentationNet representationNetwork;
    private CrossEntropyLoss policyLoss;
    private CrossEntropyLoss rewardLoss;
    private CrossEntropyLoss valueLoss;
    private CosineSimilarity cosineSimilarity;

    private Optimizer optimizer;

    public CartEfficientNet_Batch(int actionSize, int obsSize, Dictionary<string, dynamic> config) 
        : base("EfficientNet")
    {
        this.config = config;
        this.actionSize = actionSize;
        this.obsSize = obsSize;
        this.latentSize = config["latent_size"];
        this.supportWidth = config["support_width"];

        this.predictionNetwork = new CartPredictionNet(this.actionSize, this.latentSize, this.supportWidth);

        if (this.config["value_prefix"])
        {
            this.lstmHiddenSize = this.config["lstm_hidden_size"];
            this.dynamicsNetwork = new CartDynamicLSTM(
                this.actionSize,
                this.latentSize,
                this.supportWidth,
                this.lstmHiddenSize
            );
        }
        else
        {
            this.dynamicsNetworkAlternative = new CartDynamicNet(
                this.actionSize, this.latentSize, this.supportWidth
            );
        }
        this.representationNetwork = new CartRepresentationNet(this.obsSize, this.latentSize);

        this.policyLoss = nn.CrossEntropyLoss(reduction: nn.Reduction.None);
        this.rewardLoss = nn.CrossEntropyLoss(reduction: nn.Reduction.None);
        this.valueLoss = nn.CrossEntropyLoss(reduction: nn.Reduction.None);
        this.cosineSimilarity = nn.CosineSimilarity(dim: 1);

        RegisterComponents();
    }

    public Tensor ConsistencyLoss(Tensor x1, Tensor x2)
    {
        if (x1.shape != x2.shape)
        {
            throw new ArgumentException("Shapes of x1 and x2 must match.");
        }
        return -this.cosineSimilarity.forward(x1, x2);
    }

    public void InitOptimizer(double learningRate)
    {
        var parameters = new List<Parameter>(
            this.predictionNetwork.parameters().ToArray()
        );
        parameters.AddRange(this.dynamicsNetwork.parameters().ToArray());
        parameters.AddRange(this.representationNetwork.parameters().ToArray());

        this.optimizer = torch.optim.SGD(
            parameters,
            learningRate: learningRate,
            weight_decay: this.config["weight_decay"],
            momentum: this.config["momentum"]
        );
    }

    public (Tensor policy, Tensor value) Predict(Tensor latent)
    {
        return this.predictionNetwork.Forward(latent);
    }

    public (Tensor latent, Tensor valPrefix, Tensor? newHiddens) Dynamics(Tensor latent, Tensor action, (Tensor,Tensor)? hiddens = null)
    {
        if (this.config["value_prefix"])
        {
            return this.dynamicsNetwork.Forward(latent, action, hiddens!.Value);
        }
        else
        {
            var (latentResult, reward) = this.dynamicsNetworkAlternative.Forward(latent, action);
            return (latentResult, reward, null);
        }
    }

    public Tensor Represent(Tensor observation)
    {
        return this.representationNetwork.forward(observation);
    }
}
