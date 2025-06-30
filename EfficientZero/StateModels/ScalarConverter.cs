using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;
using static Google.Protobuf.Reflection.UninterpretedOption.Types;

namespace EfficientZero.Models;

public class ScalarConverter
{

    public static float SupportToScalar(Tensor support, double epsilon = 0.001)
    {
        bool shouldSqueeze = false;
        if (support.ndim == 1)
        {
            shouldSqueeze = true;
            support = support.unsqueeze(0);
        }

        int halfWidth = (int)Math.Ceiling((support.shape[1] - 1) / 2f);
        List<int> range = new List<int>();
        for (int i = -halfWidth; i <= halfWidth; i++)
            range.Add(i);

        var vals = torch.from_array(range.ToArray(), ScalarType.Float32).to(support.device);

        // Dot product of the two
        var outValue = torch.einsum("i,bi -> b", new Tensor[] { vals, support });
        var signOut = torch.where(outValue >= 0, 1, -1);

        var output = signOut * (torch.sqrt(torch.abs(outValue) + 1) - 1) + epsilon * outValue;

        var num = (torch.sqrt(1 + 4 * epsilon * (outValue.abs() + 1 + epsilon)) - 1);
        var res = (num / (2 * epsilon)).pow(2);
        var output2 = signOut * (res - 1);

        if (shouldSqueeze)
        {
            output = output.squeeze(0);
        }

        float test = output.ToSingle();
        float test2 = output2.ToSingle();

        return test;
    }

    /// <summary>
    /// Canverts a tensor array to a scalar value
    /// </summary>
    public static Tensor InverseScalarTransform(Tensor logits, double epsilon = 0.001)
    {
        //debugging
        var floatLogits = logits.data<float>().ToArray();

        var value_probs = torch.softmax(logits, dim: 1);
        var value_support = torch.ones(value_probs.shape);      
        
        int supportWidth = (int)(logits.shape[1]);
        var range = torch.ones(supportWidth, dtype:ScalarType.Float32);

        int halfWidth = (int)Math.Ceiling((supportWidth -1) / 2f);
        List<int> r = new List<int>();
        for (int i = -halfWidth; i <= halfWidth; i++)
            r.Add(i);

        //value_support[.., ..] = range;
        value_support[.., ..] = torch.from_array(r.ToArray(), ScalarType.Float32);

        value_support = value_support.to(logits.device);
        var value = (value_support * value_probs).sum(1, keepdim: true);
        var sign = torch.ones(value.shape, ScalarType.Float32).to(value.device);
        sign[value < 0] = -1.0;
        var hx = ((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon));
        var output = torch.pow(hx, 2) - 1;
        output = sign * output;

        Tensor isNan = torch.isnan(output);
        output[isNan] = 0.0;
        output[torch.abs(output) < epsilon] = 0.0;

        float scalar = value.squeeze(0).ToSingle();

        return output;
    }

    public static Tensor ScalarTransform(Tensor scalar, double epsilon = 0.001, int halfWidth = 10)
    {
        // Scaling the value function and converting to discrete support as found in
        // Appendix F of MuZero
        double scalarDbl = scalar.ToDouble();
        int signX = scalarDbl >= 0 ? 1 : -1;

        var hX = signX * (torch.sqrt(torch.abs(scalar) + 1) - 1 + epsilon * scalar);
        //var hX = torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1) - 1 + epsilon * scalar);

        hX = hX.clamp(-halfWidth, halfWidth);

        var upperNdx = (torch.ceil(hX) + halfWidth).ToInt64();
        var lowerNdx = (torch.floor(hX) + halfWidth).ToInt64();
        var ratio = hX % 1;
        var support = torch.zeros(2 * halfWidth + 1, ScalarType.Float32, device: scalar.device);

        if (upperNdx == lowerNdx)
        {
            support[upperNdx] = 1;
        }
        else
        {
            support[lowerNdx] = 1 - ratio;
            support[upperNdx] = ratio;
        }

        return support;
    }

    public static Tensor Phi(Tensor x, int min = -300, int max = 300 )
    {
        x.clamp(min, max);
        var xLow = x.floor();
        var xHigh = x.ceil();
        var pHigh = x - xLow;
        var pLow = 1 - pHigh;

        var target = torch.zeros(x.shape[0], x.shape[1]).to(x.device);
        var xHighIdx = xHigh - min;
        var xlowIdx = xLow - min;
        target.scatter(2, xHighIdx.unsqueeze(-1), pHigh.unsqueeze(-1));
        target.scatter(2, xlowIdx.unsqueeze(-1), pLow.unsqueeze(-1));

        return target;
    }
}
