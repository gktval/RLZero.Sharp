using Gym.Observations;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using nkast.Aether.Physics2D.Dynamics;
using NumSharp;
using System;

using TopDownCarPhysics.Entities;
using TopDownCarPhysics.Physics;
using TopDownCarPhysics.Utils;

namespace TopDownCarPhysics;

/// <summary>
/// This is just a simple little arcade style 2D top down car physics demo using Aether Physics engine and Monogame. It's
/// not entirely accurate (and isn't aiming to be), its not simulating each wheel or something clever like that. Think of 
/// this more as just a quick/fun car simulation that can be played about with by tweaking the physics and vehicle settings.
/// </summary>
public class GameMain : Game
{
    private GraphicsDeviceManager _graphics;
    private Opponent _opponent;
    private PhysicsWorld _physicsWorld;
    private Player _player;
    private SpriteBatch _spriteBatch;
    private RaceTrack _raceTrack;
    private GameCamera _camera;
    private FrictionDetector _contactListener;
    private BackBuffer _mainTarget;
    public Environment _env;
    private bool _canUpdate;

    public bool Render { get; set; }


    public GameMain()
    {
        _graphics = new GraphicsDeviceManager(this);
        _graphics.PreferredBackBufferWidth = 640;
        _graphics.PreferredBackBufferHeight = 640;
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        _canUpdate = true;
        Render = true;
    }

    private GraphicsDevice _gDevice;
    public GameMain(bool aiEnvironment)
    {
        _graphics = new GraphicsDeviceManager(this);
        _graphics.PreferredBackBufferWidth = 640;
        _graphics.PreferredBackBufferHeight = 640;
        PresentationParameters presParams = new PresentationParameters();
        

        Content.RootDirectory = "Content";
        IsMouseVisible = false;
        IsFixedTimeStep = true;
        TargetElapsedTime = TimeSpan.FromMilliseconds(170);

        _canUpdate = false;
        Render = false;
    }

    protected override void Initialize()
    {
        var cameraViewport = new BoxingViewportAdapter(Window, GraphicsDevice, _graphics.PreferredBackBufferWidth, _graphics.PreferredBackBufferHeight);
        _camera = new GameCamera(cameraViewport, 4000, 4000, 16);
        _camera.Zoom = .5f;

        _mainTarget = new BackBuffer(this, GraphicsDevice, _graphics.PreferredBackBufferWidth, _graphics.PreferredBackBufferHeight, DepthFormat.Depth24Stencil8);
        _mainTarget.ResetRender(this, GraphicsDevice.PresentationParameters.Bounds);

        base.Initialize();
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);

        _env = new Environment(Content, _spriteBatch);
        _contactListener = new FrictionDetector(_env);

        // We'll need a physics world to simulate some car stuff
        _physicsWorld = new PhysicsWorld();
        _physicsWorld.Gravity = Vector2.Zero;
        _physicsWorld.ContactManager.BeginContact = new BeginContactDelegate(_contactListener.BeginContact);
        _physicsWorld.ContactManager.EndContact = new EndContactDelegate(_contactListener.EndContact);

        _raceTrack = new RaceTrack(GraphicsDevice, _physicsWorld);

        bool trackCreated = false;
        while (!trackCreated)
            trackCreated = _raceTrack.CreateTrack();

        _env.Track = _raceTrack.Track;
        _env.RoadColor = new Color(102, 102, 102);// _raceTrack.RoadColor;

        // For the physics simulation to work correctly we need to indicate how many pixels
        // on the screen correspond to how many simulation units. So 'X' number of pixels
        // for 1 metre in the physics simulation. Our car is 64 pixels long, so if we say
        // the car is about 4 metres then 64/4 = 16. So 16 pixels will be 1 metre in the
        // physics simulation. The same thinking must also apply to other unit conversions!
        _physicsWorld.SetDisplayUnitToSimUnitRatio(16);

        // Create a basic player
        // 16 is tile size, 4 is scale, 2 is zoom
        Vector2 initPosition = new Vector2(_raceTrack.RoadPoly[0].Verts[0].X + (float)Math.Cos(_raceTrack.Track[0].Beta) * 16 * 4 * 2, _raceTrack.RoadPoly[0].Verts[0].Y + (float)Math.Sin(_raceTrack.Track[0].Beta) * 16 * 4 * 2);
        _player = new Player(initPosition, _raceTrack.Track[0].Beta + MathHelper.Pi / 2, _spriteBatch, _physicsWorld, Content);
        //_player = new Player(new Vector2(2000,2000), _spriteBatch, _physicsWorld, Content);

        _player.LoadContent();

        // Create an opponent
        //_opponent = new Opponent(new Vector2(150, 150), _spriteBatch, _physicsWorld, Content);
        //_opponent.LoadContent();

        // Create 'edges' for the physics engine so we don't go off screen - basically a 'box' around the screen that the
        // vehicles will 'realistically' bounce/bump off when they hit it... well, sort of ;-)
        var topLeft = new Vector2(_raceTrack.Bounds.Left, _raceTrack.Bounds.Top) * 2;
        var topRight = new Vector2(_raceTrack.Bounds.Right, _raceTrack.Bounds.Top) * 2;
        var bottomLeft = new Vector2(_raceTrack.Bounds.Left, _raceTrack.Bounds.Bottom) * 2;
        var bottomRight = new Vector2(_raceTrack.Bounds.Right, _raceTrack.Bounds.Bottom) * 2;

        _physicsWorld.CreateEdge(_physicsWorld.ToSimUnits(topLeft), _physicsWorld.ToSimUnits(topRight));
        _physicsWorld.CreateEdge(_physicsWorld.ToSimUnits(topRight), _physicsWorld.ToSimUnits(bottomRight));
        _physicsWorld.CreateEdge(_physicsWorld.ToSimUnits(bottomLeft), _physicsWorld.ToSimUnits(bottomRight));
        _physicsWorld.CreateEdge(_physicsWorld.ToSimUnits(topLeft), _physicsWorld.ToSimUnits(bottomLeft));

    }


    protected override void Update(GameTime gameTime)
    {
        _camera.Update(gameTime, _player.Position);

        if (_canUpdate)
        {
            // Get keyboard state
            var keyboardState = Keyboard.GetState();
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || keyboardState.IsKeyDown(Keys.Escape))
                Exit();

            if (keyboardState.IsKeyDown(Keys.OemOpenBrackets))
                _camera.ZoomIn(.01f);
            else if (keyboardState.IsKeyDown(Keys.OemCloseBrackets))
                _camera.ZoomOut(.01f);

            // Update the player/opponent
            _player.Update(gameTime);
            //_opponent.Update(gameTime);

            // Update the physics 'world'
            _physicsWorld.Step((float)gameTime.ElapsedGameTime.TotalSeconds);

        }

        base.Update(gameTime);
    }


    protected override void Draw(GameTime gameTime)
    {

        GraphicsDevice.SetRenderTarget(_mainTarget);
        GraphicsDevice.Clear(Color.Black);

        Matrix viewMatrix = _camera.GetViewMatrix();
        Matrix mapProjection = Matrix.CreateOrthographicOffCenter(0, _graphics.PreferredBackBufferWidth / 1f,
              _graphics.PreferredBackBufferHeight / 1f, 0, 0, -1);


        _spriteBatch.Begin(
            sortMode: SpriteSortMode.Immediate,
            blendState: null,
            samplerState: SamplerState.PointClamp,
            depthStencilState: null,
            rasterizerState: null,
            effect: null,
            transformMatrix: viewMatrix);

        //Draw the track
        _raceTrack.Render(_spriteBatch, viewMatrix, mapProjection);

        // Draw the player and the opponent
        _player.Draw();
        //_opponent.Draw();

        _spriteBatch.End();
        GraphicsDevice.SetRenderTarget(null);

        //if (Render)
        //{
        _spriteBatch.Begin(SpriteSortMode.Immediate, null, SamplerState.PointClamp);
        _spriteBatch.Draw(_mainTarget, new Rectangle(0, 0, (int)(_graphics.PreferredBackBufferWidth),
               (int)(_graphics.PreferredBackBufferHeight)), Color.White);

        //_env.Draw();

        // Finish drawing
        _spriteBatch.End();
        //}

        base.Draw(gameTime);
    }

    private TimeSpan _totalGameTime;


    public Step Step(int act)
    {
        TimeSpan elapsedTime = TimeSpan.FromSeconds(1 / 60f);
        GameTime gameTime = new GameTime(_totalGameTime, elapsedTime);
        _totalGameTime += elapsedTime;

        _player.Step(act, gameTime);

        // Update the physics 'world'
        _physicsWorld.Step((float)gameTime.ElapsedGameTime.TotalSeconds);

        bool done = _env.IsDone;
        _env.Reward -= 0.1f;

        var step_reward = _env.Reward - _env.PreviousReward;
        _env.PreviousReward = _env.Reward;
        if (_env.TileVisitedCount == _env.Track.Length || _env.NewLap)
        {
            done = true;
        }
        //Vector2 pos = Car.Hull.Position;
        //if (Math.Abs(pos.X) > PLAYFIELD || Math.Abs(pos.Y) > PLAYFIELD)
        //{
        //    done = true;
        //    step_reward = -100f;
        //}

        Step step = new Step();
        step.Done = done;
        step.Reward = step_reward;

        step.Observation = GetScreenBuffer();

        return step;
    }

    public float[,,] GetScreenBuffer()
    {

        int w = _graphics.PreferredBackBufferWidth;
        int h = _graphics.PreferredBackBufferHeight;

        int desiredWidth = 96;
        int desiredHeight = 96;

        //pull the picture from the buffer 
        int[] backBuffer = new int[w * h];

        byte[] buffer = new byte[w * h * 3];

        //copy into a texture 
        var crop = Crop(_mainTarget, new Rectangle(0, 0, desiredWidth, desiredHeight));

        backBuffer = new int[desiredWidth * desiredHeight];
        crop.GetData(backBuffer);

        float[,,] b = new float[desiredWidth,desiredHeight, 3];
        int x = 0;
        int y = 0;

        for (int i = 0; i < backBuffer.Length; i++)
        {
            if (x >= desiredWidth)
            {
                y += 1;
                x = 0;
            }

            var color = new Color((uint)backBuffer[i]);
            b[x, y, 0] = color.B / 255.0f;
            b[x, y, 1] = color.G / 255.0f;
            b[x, y, 2] = color.R / 255.0f;

            x += 1;
        }

        // For testing purpose
        //buffer = new byte[desiredWidth * desiredHeight * 3];
        //int bIndex = 0;
        //for (int i = 0; i < backBuffer.Length; i++)
        //{
        //    var color = new Color((uint)backBuffer[i]);
        //    buffer[bIndex] = color.B ;
        //    buffer[bIndex+1] = color.G ;
        //    buffer[bIndex+ 2] = color.R ;

        //    bIndex += 3;
        //}

        //var bmp = new System.Drawing.Bitmap(desiredWidth, desiredHeight);
        //var data = bmp.LockBits(new System.Drawing.Rectangle(0, 0, desiredWidth, desiredHeight), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        //System.Runtime.InteropServices.Marshal.Copy(buffer, 0, data.Scan0, buffer.Length);
        //bmp.UnlockBits(data);

        //bmp.Save(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), car_race.bmp"));

        return b;

    }

    public static Texture2D Crop(Texture2D image, Rectangle destRect)
    {
        Rectangle sourceRect = new Rectangle(0,0, image.Width, image.Height);
        var graphics = image.GraphicsDevice;
        var ret = new RenderTarget2D(graphics, destRect.Width, destRect.Height);
        var sb = new SpriteBatch(graphics);

        graphics.SetRenderTarget(ret); // draw to image
        graphics.Clear(new Color(0, 0, 0, 0));

        sb.Begin();
        sb.Draw(image, destRect, sourceRect, Color.White);//,0,Vector2.Zero,9.7916f,SpriteEffects.None, 0 );
        sb.End();

        graphics.SetRenderTarget(null); // set back to main window

        return (Texture2D)ret;
    }

    public NDArray Reset()
    {
        _env.Reset();

        _player.Tiles.Clear();

        for (int i = _physicsWorld.BodyList.Count - 1; i >= 0; i--)
        {
            var body = _physicsWorld.BodyList[i];
            if (body.Tag is RoadTile || body.Tag is Vehicle)
            {
                _physicsWorld.Remove(body);
            }
        }
        _physicsWorld.ClearForces();

        bool trackCreated = false;
        while (!trackCreated)
            trackCreated = _raceTrack.CreateTrack();

        _env.Track = _raceTrack.Track;
        _env.RoadColor = new Color(102, 102, 102);// _raceTrack.RoadColor;

        Vector2 initPosition = new Vector2(_raceTrack.RoadPoly[0].Verts[0].X + (float)Math.Cos(_raceTrack.Track[0].Beta) * 16 * 4 * 2, _raceTrack.RoadPoly[0].Verts[0].Y + (float)Math.Sin(_raceTrack.Track[0].Beta) * 16 * 4 * 2);
        _player.Reset(initPosition, _raceTrack.Track[0].Beta + MathHelper.Pi / 2);

        Step step = new Step();
        step.Observation = GetScreenBuffer();

        return step.Observation;
    }

    public void CloseEnvironment()
    {
        Exit();
    }
}
