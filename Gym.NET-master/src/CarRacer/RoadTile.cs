using Microsoft.Xna.Framework;
using nkast.Aether.Physics2D.Dynamics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TopDownCarPhysics;

public class RoadTile
{
    public float Friction { get; set; } = 0f;
    public bool RoadVisited { get; set; } = false;
    public Color Color { get; set; } = new Color(102, 102, 102); // 0.4,0.4,0.4
    public int Index { get; set; } = 0;
    public Fixture PhysicsFixture { get; set; }
}
