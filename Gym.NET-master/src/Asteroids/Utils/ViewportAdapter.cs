﻿using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asteroids_Deluxe.Utils;

public abstract class ViewportAdapter
{
    protected ViewportAdapter(GraphicsDevice graphicsDevice)
    {
        GraphicsDevice = graphicsDevice;
    }

    public GraphicsDevice GraphicsDevice { get; }
    public Viewport Viewport => GraphicsDevice.Viewport;

    public abstract int VirtualWidth { get; }
    public abstract int VirtualHeight { get; }
    public abstract int ViewportWidth { get; }
    public abstract int ViewportHeight { get; }

    public Rectangle BoundingRectangle => new Rectangle(0, 0, (int)VirtualWidth, (int)VirtualHeight);
    public Point Center => BoundingRectangle.Center;
    public abstract Matrix GetScaleMatrix();

    public Point PointToScreen(Point point)
    {
        return PointToScreen(point.X, point.Y);
    }

    public virtual Point PointToScreen(int x, int y)
    {
        var scaleMatrix = GetScaleMatrix();
        var invertedMatrix = Matrix.Invert(scaleMatrix);
        return Vector2.Transform(new Vector2(x, y), invertedMatrix).ToPoint();
    }

    public virtual void Reset()
    {
    }
}
