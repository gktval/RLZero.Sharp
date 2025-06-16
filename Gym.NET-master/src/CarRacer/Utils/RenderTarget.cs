using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TopDownCarPhysics.Utils;

public class BackBuffer : RenderTarget2D
{
    public Rectangle DestinationRectangle { get; private set; }

    public Rectangle Viewport { get; private set; }
    public Vector2 ViewportTopLeft { get { return new Vector2(Viewport.X, Viewport.Y); } }

    public BackBuffer(Game game, GraphicsDevice graphicsDevice, int width, int height, DepthFormat depthFormat)
        : base(graphicsDevice, width, height, false,game.GraphicsDevice.PresentationParameters.BackBufferFormat, depthFormat)
    {
        CalculateRenderingMetrics(game, graphicsDevice.PresentationParameters.Bounds);
    }
    public void ResetRender(Game game, Rectangle screenArea)
    {
        CalculateRenderingMetrics(game, screenArea);
    }

    private void CalculateRenderingMetrics(Game game, Rectangle screenArea)
    {
        float horizontalRatio = (float)screenArea.Width / (float)this.Width;
        float scaledHeight = (float)this.Height * horizontalRatio;
        float scaledVerticalOverspill = (scaledHeight - screenArea.Height) / 2.0f;
        float visibleVerticalArea = this.Height - ((scaledVerticalOverspill / horizontalRatio) * 2.0f);

        //MapRenderer mapRenderer = game.Services.GetService<MapRenderer>();
        //TiledMap map = mapRenderer.Map;
        //Vector2 offset = Vector2.Zero;
        //Camera camera = game.Services.GetService<Camera>();
        //if (GameConfig.GuiWidth > map.Width * map.TileWidth * 2)
        //{
        //    //Matrix translation = Matrix.CreateTranslation((GameConfig.GuiWidth - map.Width * map.TileWidth * 2) / 4f, 0, 0);
        //    offset.X = (GameConfig.GuiWidth / 2) - (map.Width * map.TileHeight * 2) / 2f;
        //}
        //if (GameConfig.GuiHeight > map.Height * map.TileHeight * 2)
        //{
        //    //Matrix translation = Matrix.CreateTranslation((GameConfig.GuiWidth - map.Width * map.TileWidth * 2) / 4f, 0, 0);
        //    offset.Y = (GameConfig.GuiHeight / 2) - (map.Height * map.TileHeight * 2) / 2f;
        //}

        //DestinationRectangle = new Rectangle((int)offset.X, (int)offset.Y - (int)scaledVerticalOverspill, screenArea.Width, (int)scaledHeight);

        DestinationRectangle = new Rectangle(0, 0, screenArea.Width, screenArea.Height);

        Viewport = new Rectangle(0, (int)(scaledVerticalOverspill / horizontalRatio), this.Width, (int)Math.Round(visibleVerticalArea, 0));
    }
}
