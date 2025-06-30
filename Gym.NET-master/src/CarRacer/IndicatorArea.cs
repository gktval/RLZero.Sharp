using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System;
using TopDownCarPhysics.Entities;

namespace TopDownCarPhysics;

internal class IndicatorArea
{
    private GameMain _game;
    private GraphicsDeviceManager _graphics;
    private readonly Texture2D _texture;
    private SpriteFont _font;

    public IndicatorArea(GameMain game, GraphicsDeviceManager graphics)
    {
        _game = game;
        _graphics = graphics;
        _texture = new Texture2D(_graphics.GraphicsDevice, 40, 40);
    }

    internal void LoadContent(ContentManager cm)
    {
        // Load a font
        _font = cm.Load<SpriteFont>("font");
    }

    private Vector2[] GetIndicator(float place, float s, float l, float w, float h, float v, bool vertical = true)
    {
        Vector2[] path = null;
        if (vertical)
        {
            path = new Vector2[] {
                    new Vector2(place*s     ,h-(l+l*v)),
                    new Vector2((place+1f)*s,h-(l+l*v)),
                    new Vector2((place+1f)*s,h-l),
                    new Vector2(place*s     ,h-l),
                };
        }
        else
        {
            path = new Vector2[] {
                    new Vector2(place*s    ,h-4*l),
                    new Vector2((place+v)*s,h-4*l),
                    new Vector2((place+v)*s,h-1*l),
                    new Vector2(place*s    ,h-1*l),
                };
        }
        return (path);
    }

    internal void RenderIndicators(Player player, SpriteBatch spriteBatch, float score)
    {
        CreateBorder(_texture, 1, Color.Black);

        int w = _graphics.PreferredBackBufferWidth;
        int h = _graphics.PreferredBackBufferHeight;
        float s = w / 40f;
        float l = h / 40f;
        Color color = new Color(0, 0, 0);
        Vector2[] poly = new Vector2[] {
                new Vector2(w,h),
                new Vector2(w,h-5f*l),
                new Vector2(0f,h-5f*l),
                new Vector2(0f,h)
            };

        //Draw the background
        Rectangle destRect = new Rectangle(0, (int)(_graphics.PreferredBackBufferHeight * 7 / 8f), w, h / 8);
        spriteBatch.Draw(_texture, destRect, null,
                            Color.White, 0f, Vector2.Zero, SpriteEffects.None, 0.65f);

        // Draw the telemetry indicators
        float v = player.LinearVelocity.Length();
        Vector2[] coords = GetIndicator(10f, s, l, w, h, 0.08f * v);
        DrawIndicator(spriteBatch, _texture, coords, Color.White);

        v = System.Math.Abs(player.LinearDamping - 0.5f);
        coords = GetIndicator(12f, s, l, w, h, 6f * v);
        DrawIndicator(spriteBatch, _texture, coords, Color.Purple);

        Vector2 forward = player.Forward * player.LinearVelocity;
        v = System.Math.Abs(forward.Length());
        coords = GetIndicator(20f, s, l, w, h, .28f * v, false);
        DrawIndicator(spriteBatch, _texture, coords, forward.X < 0 && forward.Y < 0 ? Color.Green : Color.Yellow);

        v = System.Math.Abs(player.AngularVelocity);
        coords = GetIndicator(30f, s, l, w, h, 1.6f * v, false);
        DrawIndicator(spriteBatch, _texture, coords, player.AngularVelocity > 0 ? Color.Red : Color.Blue);

        spriteBatch.DrawString(_font, Math.Round(score, 1).ToString(), new Vector2(20f, h - 54), Color.White,0f,Vector2.Zero,2f,SpriteEffects.None,1f);
    }

    private void CreateBorder(Texture2D texture, int borderWidth, Color borderColor)
    {
        Color[] colors = new Color[texture.Width * texture.Height];

        for (int x = 0; x < texture.Width; x++)
        {
            for (int y = 0; y < texture.Height; y++)
            {
                for (int i = 0; i <= borderWidth; i++)
                {
                    colors[x + y * texture.Width] = borderColor;
                }
            }
        }

        texture.SetData(colors);
    }

    private void DrawIndicator(SpriteBatch spriteBatch, Texture2D texture, Vector2[] points, Color borderColor)
    {
        Color[] colors = new Color[texture.Width * texture.Height];

        for (int x = 0; x < texture.Width; x++)
        {
            for (int y = 0; y < texture.Height; y++)
            {
                for (int i = 0; i <= points.Length; i++)
                {
                    colors[x + y * texture.Width] = borderColor;
                }
            }
        }

        texture.SetData(colors);

        int w = (int)(points[1].X - points[0].X);
        int h = (int)(points[2].Y - points[1].Y);
        Rectangle destRect = new Rectangle((int)points[0].X, (int)points[0].Y, w, h);
        spriteBatch.Draw(_texture, destRect, null,
                            Color.White, 0f, Vector2.Zero, SpriteEffects.None, 0.65f);
    }
}
