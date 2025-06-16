using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace EfficientZero.Models;

public class CosineSimilarity:nn.Module
{
    public long dim { get; set; }

    public double eps { get; set; }

    public CosineSimilarity(long dim = 1L, double eps = 1E-08)
        : base("CosineSimilarity")
    {
        this.dim = dim;
        this.eps = eps;
    }

    public Tensor Forward(torch.Tensor input1, torch.Tensor input2)
    {
        return torch.nn.functional.cosine_similarity(input1, input2, dim, eps);
    }
}
