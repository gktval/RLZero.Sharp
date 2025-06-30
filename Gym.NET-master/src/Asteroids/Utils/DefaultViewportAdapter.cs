using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asteroids_Deluxe.Utils;

public class DefaultViewportAdapter : ViewportAdapter
{
    private readonly GraphicsDevice _graphicsDevice;

    public DefaultViewportAdapter(GraphicsDevice graphicsDevice)
        : base(graphicsDevice)
    {
        _graphicsDevice = graphicsDevice;
    }

    public override int VirtualWidth => _graphicsDevice.Viewport.Width;
    public override int VirtualHeight => _graphicsDevice.Viewport.Height;
    public override int ViewportWidth => _graphicsDevice.Viewport.Width;
    public override int ViewportHeight => _graphicsDevice.Viewport.Height;

    public override Matrix GetScaleMatrix()
    {
        return Matrix.Identity;
    }
}
