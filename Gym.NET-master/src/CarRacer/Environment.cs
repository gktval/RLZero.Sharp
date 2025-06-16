using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;
namespace TopDownCarPhysics;

public class Environment
{
    private SpriteFont _font;
    private SpriteBatch _spriteBatch;
    public Environment(ContentManager content, SpriteBatch spriteBatch)
    {
        _font = content.Load<SpriteFont>("font");
        _spriteBatch = spriteBatch;
    }

    public void Reset()
    {
        Reward = 0;
        PreviousReward = 0;
        TileVisitedCount = 0;
        NewLap = false;
        LapCompletePercent = 100f;
        IsDone = false;
    }

    public float Reward { get; set; }
    public float PreviousReward { get; set; }
    public int TileVisitedCount { get; set; }
    public Color RoadColor { get; internal set; }
    public float LapCompletePercent { get; set; } = 100f;
    public Track[] Track { get; set; }
    public bool NewLap { get; internal set; }
    public bool IsDone { get; internal set; }

    public void Draw()
    {
        _spriteBatch.DrawString(_font, "Reward: " + Math.Round(Reward, 2), new Vector2(8, 10), Color.White);
    }
}
