using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero.ImageModels;

public class MLP :  Sequential<Tensor, Tensor>
{
    public MLP(int inputSize, int hiddenSizes, int outputSize, Device device,
        nn.Module<Tensor, Tensor> outputActivation = null, nn.Module<Tensor, Tensor>? activation = null, bool initZero = false)
       : base(new nn.Module<Tensor, Tensor>[] { })
    {

        if (outputActivation == null)
            outputActivation = nn.Identity(); // Identity

        if (activation == null)
            activation = nn.ELU(); // ELU or any default activation

        var sizes = new List<int> { inputSize }.Concat(new int[] { hiddenSizes }).Append(outputSize).ToList();
        var layers = new List<nn.Module<Tensor, Tensor>>();

        for (int i = 0; i < sizes.Count-1; i++)
        {
            if (i < sizes.Count - 2)
            {

                layers.Add(nn.Linear(sizes[i], sizes[i + 1], device: device));
                layers.Add(nn.BatchNorm1d(sizes[i + 1], device: device));
                layers.Add(activation);
            }
            else
            {
                layers.Add(nn.Linear(sizes[i], sizes[i + 1], device: device));
                layers.Add(outputActivation);
            }
        }

        if (initZero)
        {
            var lastLayer = (Linear)layers[layers.Count - 2];
            using (torch.no_grad())
            {
                lastLayer.weight.fill_(0);
                lastLayer.bias.fill_(0);
            }
        }

        List<string> names = new List<string>();
        foreach (var layer in layers)
        {
            string lName = layer.GetName();
            int index = 0;
            lName = lName + "_" + index;
            while (names.Contains(lName))
            {
                string[] nSplit = lName.Split('_');
                index = int.Parse(nSplit[1]) + 1;
                lName = lName + "_" + index;
            }
            names.Add(lName);

            base.Add(lName, layer);
        }
        
    }

    public override Tensor forward(Tensor tensor)
    {
        using var _ = torch.NewDisposeScope();

        var result = tensor.alias();

        foreach (var layer in children())
        {
            switch (layer)
            {
                case torch.nn.Module<torch.Tensor, torch.Tensor> m:
                    result = m.call(result);
                    break;
                default:
                    throw new InvalidCastException($"Invalid module type in MLP.");
            }
        }

        return result.MoveToOuterDisposeScope();
    }

}

