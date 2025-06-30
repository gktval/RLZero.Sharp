using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using nkast.Aether.Physics2D.Collision.Shapes;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using nkast.Aether.Physics2D.Dynamics.Contacts;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using TopDownCarPhysics.Entities;
using static System.Runtime.InteropServices.JavaScript.JSType;


namespace TopDownCarPhysics;

public class RaceTrack
{
    public const float ZOOM = 1f; // Camera zoom

    public const float SCALE = 4; // Track scale
    public const float TRACK_RAD = 600f / SCALE;
    public const int PLAYFIELD = 4000;
    public const int TILE_SIZE = 16;

    public const float TRACK_DETAIL_STEP = 21f / SCALE;
    public const float TRACK_TURN_RATE = 0.31f;
    public const float TRACK_WIDTH = 32f / SCALE;
    public const float BORDER = 4f / SCALE;
    public const int BORDER_MIN_COUNT = 4;
    public const int GRASS_DIM = (int)(PLAYFIELD / 16f);

    public Color RoadColor { get; set; } = new Color(255, 255, 255);
    public Color BackgroundColor { get; set; } = new Color(102, 204, 102);
    public Color GrassColor { get; set; } = new Color(102, 230, 102);

    private World _world;
    public List<RoadPolygon> RoadPoly { get; set; }
    public List<Body> Road { get; set; }
    public Track[] Track { get; set; }
    private float StartAlpha { get; set; } = 0f;
    public Rectangle Bounds { get; set; }
    public struct RoadPolygon
    {
        internal Vertices Verts;
        internal Color Color;
        internal int TrackIndex;
        internal int RoadIndex;
    }


    private GraphicsDevice _graphicsDevice;
    private BasicEffect _basicEffect;
    public RaceTrack(GraphicsDevice graphicsDevice, Physics.PhysicsWorld physicsWorld)
    {
        _graphicsDevice = graphicsDevice;
        _world = physicsWorld;
        Bounds = new Rectangle(0, 0, PLAYFIELD, PLAYFIELD);

        float bounds = PLAYFIELD;
        _backgroundTexture = new Texture2D(graphicsDevice, (int)bounds, (int)bounds);
        _grassTexture = new Texture2D(graphicsDevice, 16, 16);
        _roadTexture = new Texture2D(graphicsDevice, 16, 16);

        CreateTile(_backgroundTexture, BackgroundColor);
        CreateTile(_grassTexture, GrassColor);
        CreateTile(_roadTexture, RoadColor);

        _basicEffect = new BasicEffect(graphicsDevice);
        _basicEffect.Texture = _roadTexture;
        _basicEffect.TextureEnabled = true;

    }

    public void Render(SpriteBatch spriteBatch, Matrix viewMatrix, Matrix mapProjection)
    {
        float T = 1f;
        float angle = 0;// -Car.Hull.Rotation;
        // Animating first second zoom.
        float zoom = 0.1f * SCALE * (float)Math.Max(1f - T, 0f) + ZOOM * SCALE * (float)Math.Min(T, 1f);
        //float scroll_x = -(Car.Hull.Position.X) * zoom;
        //float scroll_y = -(Car.Hull.Position.Y) * zoom;
        //Vector2 trans = CarDynamics.RotateVec(new Vector2(scroll_x, scroll_y), angle) + new Vector2(WINDOW_W / 2, WINDOW_H / 4);
        Vector2 trans = new Vector2(0, 0);
        RenderRoad(spriteBatch, zoom, trans, angle, viewMatrix, mapProjection);
    }

    private Texture2D _backgroundTexture;
    private Texture2D _grassTexture;
    private Texture2D _roadTexture;
    private void RenderRoad(SpriteBatch spriteBatch, float zoom, Vector2 trans, float angle, Matrix viewMatrix, Matrix mapProjection)
    {
        int bounds = PLAYFIELD;
        var field = new Rectangle(0, 0, bounds * 2, bounds * 2);

        spriteBatch.Draw(_backgroundTexture, field, null,
                            Color.White, 0f, Vector2.Zero, SpriteEffects.None, 1f);

        int k = GRASS_DIM;


        for (int x = 0; x < 32; x += 2)
        {
            for (int y = 0; y < 32; y += 2)
            {
                Rectangle poly = new Rectangle(k * x + k, k * y, k, k);

                spriteBatch.Draw(_grassTexture, poly, null,
                            Color.White, 0f, Vector2.Zero, SpriteEffects.None, 1f);
            }
        }

        short[] ind = new short[6];
        ind[0] = 0;
        ind[1] = 2;
        ind[2] = 1;
        ind[3] = 0;
        ind[4] = 3;
        ind[5] = 2;

        _basicEffect.View = viewMatrix;
        _basicEffect.World = Matrix.Identity;
        _basicEffect.Projection = mapProjection;
        _basicEffect.VertexColorEnabled = true;

        for (int i = 0; i < RoadPoly.Count; i++)
        {
            VertexPositionColorTexture[] vert = new VertexPositionColorTexture[4];

            vert[0].Position = new Vector3(RoadPoly[i].Verts[0].X, RoadPoly[i].Verts[0].Y, 0);
            vert[1].Position = new Vector3(RoadPoly[i].Verts[1].X, RoadPoly[i].Verts[1].Y, 0);
            vert[2].Position = new Vector3(RoadPoly[i].Verts[2].X, RoadPoly[i].Verts[2].Y, 0);
            vert[3].Position = new Vector3(RoadPoly[i].Verts[3].X, RoadPoly[i].Verts[3].Y, 0);

            vert[0].TextureCoordinate = new Vector2(0, 0);
            vert[1].TextureCoordinate = new Vector2(1, 0);
            vert[2].TextureCoordinate = new Vector2(0, 1);
            vert[3].TextureCoordinate = new Vector2(1, 1);

            vert[0].Color = RoadPoly[i].Color;
            vert[1].Color = RoadPoly[i].Color;
            vert[2].Color = RoadPoly[i].Color;
            vert[3].Color = RoadPoly[i].Color;

            foreach (EffectPass effectPass in _basicEffect.CurrentTechnique.Passes)
            {
                effectPass.Apply();

                _graphicsDevice.DrawUserIndexedPrimitives<VertexPositionColorTexture>(
PrimitiveType.TriangleList, vert, 0, vert.Length, ind, 0, ind.Length / 3);

            }
        }
    }

    public static void CreateTile(Texture2D texture, Color tileColor)
    {
        Color[] colors = new Color[texture.Width * texture.Height];

        for (int x = 0; x < texture.Width; x++)
        {
            for (int y = 0; y < texture.Height; y++)
            {
                colors[x + y * texture.Width] = tileColor;
            }
        }

        texture.SetData(colors);
    }

    public bool CreateTrack()
    {
        Road = new List<Body>();
        RoadPoly = new List<RoadPolygon>();

        int CHECKPOINTS = 12;
        float[] checkpoints = new float[CHECKPOINTS * 3];
        float frac = 2f * (float)Math.PI / (float)CHECKPOINTS;

        Random random = new Random();
        // Create checkpoints
        int i = 0;
        for (i = 0; i < CHECKPOINTS; i++)
        {
            int j = i * 3; // index into the checkpoints array
            float noise = random.NextSingle() * frac;// RandomState.uniform(0f, frac);
            float alpha = frac * (float)i + noise;
            float minTrack_rad = TRACK_RAD / 3f;
            float rad = random.NextSingle() * (TRACK_RAD - minTrack_rad) + minTrack_rad;

            if (i == 0)
            {
                alpha = 0f;
                rad = 1.5f * TRACK_RAD;
            }
            else if (i == CHECKPOINTS - 1)
            {
                alpha = frac * (float)i;
                StartAlpha = frac * (-0.5f);
                rad = 1.5f * TRACK_RAD;
            }
            checkpoints[j++] = alpha;
            checkpoints[j++] = rad * (float)Math.Cos(alpha);
            checkpoints[j++] = rad * (float)Math.Sin(alpha);
        }
        Road.Clear();
        // Go from one checkpoint to another to create the xtrack.
        float x = 1.5f * TRACK_RAD;
        float y = 0f;
        float beta = 0f;
        int dest_i = 0;
        int laps = 0;
        int no_freeze = 2500;
        bool visited_other_size = false;
        List<Track> xtrack = new List<Track>();
        while (no_freeze > 0)
        {
            float alpha = (float)Math.Atan2(y, x);
            if (visited_other_size && alpha > 0f)
            {
                laps++;
                visited_other_size = false;
            }
            if (alpha < 0f)
            {
                visited_other_size = true;
                alpha += 2f * (float)Math.PI;
            }
            float dest_x = 0f;
            float dest_y = 0f;
            bool failed = false;
            while (true)
            {
                failed = true;
                while (true)
                {
                    float dest_alpha = checkpoints[dest_i % checkpoints.Length];
                    dest_x = checkpoints[dest_i % checkpoints.Length + 1];
                    dest_y = checkpoints[dest_i % checkpoints.Length + 2];
                    if (alpha <= dest_alpha)
                    {
                        failed = false;
                        break;
                    }
                    dest_i += 3; // each element is 3
                    if (dest_i % checkpoints.Length == 0)
                        break;
                }
                if (!failed)
                {
                    break;
                }
                alpha -= 2f * (float)Math.PI;
            }

            float r1x = (float)Math.Cos(beta);
            float r1y = (float)Math.Sin(beta);
            float p1x = -r1y;
            float p1y = r1x;
            float dest_dx = dest_x - x; // Vector towards destination
            float dest_dy = dest_y - y;
            // Destination vector projected on rad:
            float proj = r1x * dest_dx + r1y * dest_dy;
            while ((beta - alpha) > (1.5f * (float)Math.PI))
            {
                beta -= 2f * (float)Math.PI;
            }
            while ((beta - alpha) < (1.5f * (float)Math.PI))
            {
                beta += 2f * (float)Math.PI;
            }
            float prev_beta = beta;
            proj *= SCALE;
            if (proj > 0.3f)
                beta -= Math.Min(TRACK_TURN_RATE, (float)Math.Abs(0.001 * proj));
            if (proj < -0.3f)
                beta += Math.Min(TRACK_TURN_RATE, (float)Math.Abs(0.001 * proj));
            x += p1x * TRACK_DETAIL_STEP;
            y += p1y * TRACK_DETAIL_STEP;
            xtrack.Add(new Track(alpha, prev_beta * 0.5f + beta * 0.5f, x, y));

            if (laps > 1)
            {
                break;
            }
            no_freeze--;
        }

        // Find closed loop range i1 .. i2, first loop should be ignored, second is OK
        int i1 = -1;
        int i2 = -1;
        i = xtrack.Count;
        while (true)
        {
            i -= 1;
            if (i == 0)
            {
                // Failed
                return (false);
            }
            bool pass_through_start = (xtrack[i].Alpha > StartAlpha && xtrack[i - 1].Alpha <= StartAlpha);
            if (pass_through_start && i2 == -1)
            {
                i2 = i;
            }
            else if (pass_through_start && i1 == -1)
            {
                i1 = i;
                break;
            }
        }
        //if (Verbose)
        //{
        Debug.WriteLine("Track generation {0}..{1} -> {2}-tiles track", i1, i2, i2 - i1);
        //}
        Debug.Assert(i1 != -1);
        Debug.Assert(i2 != -1);
        Track[] track = new Track[(i2 - i1)];
        Array.Copy(xtrack.ToArray().Skip(i1).ToArray(), track, i2 - i1);

        float first_beta = xtrack[0].Beta;
        float first_perp_x = (float)Math.Cos(first_beta);
        float first_perp_y = (float)Math.Sin(first_beta);
        // Length of perpendicular jump to put together head and tail
        float a = first_perp_x * (track[0].X - track[track.Length - 1].X);
        float b = first_perp_y * (track[0].Y - track[track.Length - 1].Y);
        float well_glued_together = (float)Math.Sqrt(a * a + b * b);
        if (well_glued_together > TRACK_DETAIL_STEP)
        {
            return false;
        }
        // Red-white border on hard turns
        bool[] border = new bool[track.Length];
        for (i = 0; i < track.Length; i += 1)
        {
            bool good = true;
            int oneside = 0;
            for (int neg = 0; neg < BORDER_MIN_COUNT; neg++)
            {
                int idx = i - neg;
                while (idx < 0)
                {
                    idx += track.Length;
                }
                float beta1 = track[idx].Beta;
                idx = i - neg - 1;
                while (idx < 0)
                {
                    idx += track.Length;
                }
                float beta2 = track[idx].Beta; // index out of bounds TODO!
                float dbeta = beta1 - beta2;
                good &= (Math.Abs(dbeta) > TRACK_TURN_RATE * 0.2f);
                oneside += (dbeta < 0f ? -1 : (dbeta > 0f ? 1 : 0));
            }
            good = good && (Math.Abs(oneside) == BORDER_MIN_COUNT);
            border[i] = good;
        }
        for (i = 0; i < border.Length; i++)
        {
            for (int neg = 0; neg < BORDER_MIN_COUNT; neg++)
            {
                int j = i - neg;
                if (j < 0)
                {
                    j += border.Length;
                }

                border[j] |= border[i];
            }
        }

        int offset = (int)(TRACK_RAD * SCALE);
        Vector2 add = new Vector2(TRACK_RAD + TRACK_RAD / 2, TRACK_RAD + TRACK_RAD / 2);

        // Create tiles
        int prev_track_index = track.Length - 1;
        int roadIndex = 0;
        for (i = 0; i < track.Length; i++)
        {
            // position 1
            float beta1 = track[i].Beta;
            float x1 = track[i].X;
            float y1 = track[i].Y;
            float cos_beta1 = (float)Math.Cos(beta1);
            float sin_beta1 = (float)Math.Sin(beta1);
            // previous position
            float beta2 = track[prev_track_index].Beta;
            float x2 = track[prev_track_index].X;
            float y2 = track[prev_track_index].Y;
            float cos_beta2 = (float)Math.Cos(beta2);
            float sin_beta2 = (float)Math.Sin(beta2);
            // Polygon
            Vector2 road1_l = new Vector2(x1 - TRACK_WIDTH * cos_beta1, y1 - TRACK_WIDTH * sin_beta1);
            Vector2 road1_r = new Vector2(x1 + TRACK_WIDTH * cos_beta1, y1 + TRACK_WIDTH * sin_beta1);
            Vector2 road2_l = new Vector2(x2 - TRACK_WIDTH * cos_beta2, y2 - TRACK_WIDTH * sin_beta2);
            Vector2 road2_r = new Vector2(x2 + TRACK_WIDTH * cos_beta2, y2 + TRACK_WIDTH * sin_beta2);
            Vertices vx = new Vertices();
            vx.Add(road1_l + add);
            vx.Add(road1_r + add);
            vx.Add(road2_r + add);
            vx.Add(road2_l + add);
            // fixtureDef(shape = polygonShape(vertices =[(0, 0), (1, 0), (1, -1), (0, -1)]))
            Body bx = _world.CreateBody(bodyType: BodyType.Static);
            Fixture t = bx.CreateFixture(new PolygonShape(vx, 1f));
            RoadTile tile = new RoadTile();
            bx.Tag = tile;
            tile.Index = i;
            int c = 10 * (tile.Index % 3);
            tile.Color = new Color(102 + c, 102 + c, 102 + c);
            tile.RoadVisited = false;
            tile.Friction = 1f;
            t.IsSensor = true;
            tile.PhysicsFixture = t;
            RoadPolygon poly = new RoadPolygon();
            poly.Verts = vx;
            poly.Color = tile.Color;
            poly.TrackIndex = tile.Index;
            poly.RoadIndex = roadIndex;
            roadIndex++;
            RoadPoly.Add(poly);
            Road.Add(bx);
            if (border[tile.Index])
            {
                int side = ((beta2 - beta1) < 0) ? -1 : 1;
                Vector2 b1_l = new Vector2(x1 + side * TRACK_WIDTH * cos_beta1, y1 + side * TRACK_WIDTH * sin_beta1);
                Vector2 b1_r = new Vector2(x1 + side * (TRACK_WIDTH + BORDER) * cos_beta1, y1 + side * (TRACK_WIDTH + BORDER) * sin_beta1);
                Vector2 b2_l = new Vector2(x2 + side * TRACK_WIDTH * cos_beta2, y2 + side * TRACK_WIDTH * sin_beta2);
                Vector2 b2_r = new Vector2(x2 + side * (TRACK_WIDTH + BORDER) * cos_beta2, y2 + side * (TRACK_WIDTH + BORDER) * sin_beta2);
                vx = new Vertices();
                if (side == 1)
                {
                    vx.Add(b1_l + add);
                    vx.Add(b1_r + add);
                    vx.Add(b2_r + add);
                    vx.Add(b2_l + add);
                }
                else
                {
                    vx.Add(b1_r + add);
                    vx.Add(b1_l + add);
                    vx.Add(b2_l + add);
                    vx.Add(b2_r + add);
                }
                poly = new RoadPolygon();
                poly.Verts = vx;
                poly.Color = (tile.Index % 2 == 0) ? new Color(255, 255, 255) : new Color(255, 0, 0);
                poly.TrackIndex = -1;
                poly.RoadIndex = roadIndex;
                roadIndex++;
                RoadPoly.Add(poly);
            }

            prev_track_index = (prev_track_index + 1) % track.Length;
        }

        foreach (RoadPolygon poly in RoadPoly)
        {
            for (int v = 0; v < poly.Verts.Count; v++)
            {
                poly.Verts[v] = poly.Verts[v] * TILE_SIZE;
            }
        }

        Track = track;
        return true;
    }

}
