using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TopDownCarPhysics;

public class Track
{

    public Track(float alpha, float beta, float x, float y)
    {
        Alpha = alpha;
        Beta = beta;
        X = x;
        Y = y;
    }

    public float Alpha { get; set; }
    public float Beta { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}
