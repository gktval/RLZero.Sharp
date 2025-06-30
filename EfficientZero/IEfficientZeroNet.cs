using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace EfficientZero;

public interface IEfficientZeroNet
{
    public void Eval();
    public void Train();
    public void Save(string location);
    public void Load(string location, bool strict = true);
    public void InitOptimizer(double learningRate);
    public (Tensor state, Tensor value, Tensor policy) InitialInference(Tensor observation);


    public (Tensor nextLatentState, Tensor valuePrefix, Tensor value, Tensor policy, (Tensor,Tensor) rewardHidden)
        RecurrentInference(Tensor latentState, Tensor action, (Tensor, Tensor) rewardHidden);

    public Tensor Represent(Tensor observation);

    public Tensor ConsistencyLoss(Tensor x1, Tensor x2);

    public Tensor Projection(Tensor latent, bool withGrad = true);

    public IEnumerable<Parameter> parameters(bool recurse = true);
    public Optimizer Optimizer { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> PolicyLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> RewardLoss { get; set; }
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> ValueLoss { get; set; }
}
