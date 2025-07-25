﻿using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Audio;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using System.Reflection;
using System;
using Gym.Observations;
using NumSharp;
using Asteroids_Deluxe.Utils;

namespace Asteroids_Deluxe;

using Serv = VectorEngine.Services;
using Timer = VectorEngine.Timer;
using PO = VectorEngine.PositionedObject;
/// <summary>
/// After 40,000 points only small UFOs spawn.
/// A steadily decreasing timer that shortens intervals between saucer spawns on each UFO.
/// </summary>
public class AsteroidGame : Game
{
    GraphicsDeviceManager _graphics;

    Player _player;
    PO m_PlayerClear;
    UFO m_UFO;
    PodGroup m_Pod;
    Timer m_UFOTimer;
    Timer m_PodTimer;
    Timer m_BackGroundPlay;
    Timer m_BackTwoPlay;
    Timer m_BackgroundDelay;
    Timer m_BackgroundTwoDelay;
    Timer m_GameWavePlayTime;
    List<Rock> m_Rocks;
    SoundEffect m_RockExplode;
    SoundEffect m_Background;
    Word m_AtariHUD;
    Number m_AtariDate;
    KeyboardState m_KeyState, m_KeyStateOld;
    readonly float m_UFOTimerSeedAmount = 10.15f;
    readonly float m_PodTimerSeedAmount = 30;
    readonly float m_BackgroundOneDelaySeed = 1;
    readonly float m_BackgroundTwoDelaySeed = 2;
    int m_UFOCount;
    int m_Wave;
    int m_LargeRockSpawnAmount = 2;
    int m_RockCount;
    bool m_PlayedBack = true;
    //bool m_PlayedTwo = false;
    bool m_Paused = false;
    bool _canStep = false;


    public bool IsAIEnv { get; set; }
    public bool Render { get; set; }
    public bool PlayBack
    {
        get { return m_PlayedBack; }
        set
        {
            m_PlayedBack = value;
            _player.PlayBack = value;
            m_UFO.Playback = value;
        }
    }

    public AsteroidGame(bool aiEnvironment)
    {
        _graphics = new GraphicsDeviceManager(this);
        _graphics.PreferredBackBufferWidth = 640;
        _graphics.PreferredBackBufferHeight = 640;
        PresentationParameters presParams = new PresentationParameters();

        _player = new Player(this);
        m_PlayerClear = new PO(this);
        m_UFO = new UFO(this);
        m_Pod = new PodGroup(this);
        m_UFOTimer = new Timer(this);
        m_PodTimer = new Timer(this);
        m_BackGroundPlay = new Timer(this);
        m_BackTwoPlay = new Timer(this);
        m_BackgroundDelay = new Timer(this);
        m_BackgroundTwoDelay = new Timer(this);
        m_GameWavePlayTime = new Timer(this);
        m_AtariHUD = new Word(this);
        m_AtariDate = new Number(this);
        m_Rocks = new List<Rock>();
        Content.RootDirectory = "Content";

        IsFixedTimeStep = true;
        TargetElapsedTime = TimeSpan.FromMilliseconds(170);

        IsAIEnv = aiEnvironment;
        Render = false;
        PlayBack = false;
    }

    public AsteroidGame()
    {
        _graphics = new GraphicsDeviceManager(this);
        _graphics.SynchronizeWithVerticalRetrace = true;
        //_graphics.GraphicsProfile = GraphicsProfile.HiDef;
        _graphics.PreferredBackBufferWidth = 1200;
        _graphics.PreferredBackBufferHeight = 900;
        _graphics.PreparingDeviceSettings += SetMultiSampling;
        _graphics.ApplyChanges();
        IsFixedTimeStep = false;
        _player = new Player(this);
        m_PlayerClear = new PO(this);
        m_UFO = new UFO(this);
        m_Pod = new PodGroup(this);
        m_UFOTimer = new Timer(this);
        m_PodTimer = new Timer(this);
        m_BackGroundPlay = new Timer(this);
        m_BackTwoPlay = new Timer(this);
        m_BackgroundDelay = new Timer(this);
        m_BackgroundTwoDelay = new Timer(this);
        m_GameWavePlayTime = new Timer(this);
        m_AtariHUD = new Word(this);
        m_AtariDate = new Number(this);
        m_Rocks = new List<Rock>();
        Content.RootDirectory = "Content";

        _canStep = true;
    }

    private void SetMultiSampling(object sender, PreparingDeviceSettingsEventArgs eventArgs)
    {
        PresentationParameters PresentParm = eventArgs.GraphicsDeviceInformation.PresentationParameters;
        PresentParm.MultiSampleCount = 4;
    }
    /// <summary>
    /// Allows the game to perform any initialization it needs to before starting to run.
    /// This is where it can query for any required services and load any non-graphic
    /// related content.  Calling base.Initialize will enumerate through any components
    /// and initialize them as well.
    /// </summary>
    protected override void Initialize()
    {
        Serv.Initialize(_graphics, this);

        base.Initialize();
    }

    /// <summary>
    /// LoadContent will be called once per game and is the place to load
    /// all of your content.
    /// </summary>
    protected override void LoadContent()
    {
        _player.LoadSounds(Content.Load<SoundEffect>("AsteroidsDeluxePlayerFire"),
            Content.Load<SoundEffect>("AsteroidsPlayerExplosion"),
            Content.Load<SoundEffect>("AsteroidsDeluxeBonusShip"),
            Content.Load<SoundEffect>("AsteroidsDeluxePlayerThrust"),
            Content.Load<SoundEffect>("AsteroidsDeluxePlayerStart"),
            Content.Load<SoundEffect>("AsteroidsDeluxeShield"));

        m_UFO.LoadSounds(Content.Load<SoundEffect>("AsteroidsUFOExplosion"),
            Content.Load<SoundEffect>("AsteroidsUFOShot"), Content.Load<SoundEffect>("AsteroidsUFOLarge"),
            Content.Load<SoundEffect>("AsteroidsUFOSmall"));

        m_RockExplode = Content.Load<SoundEffect>("AsteroidsRockExplosion");
        m_Background = Content.Load<SoundEffect>("AsteroidsDeluxeBackground");
        m_Pod.LoadSounds(Content.Load<SoundEffect>("AsteroidsDeluxePodSpawn"),
            Content.Load<SoundEffect>("AsteroidsDeluxePodExplosion"));

        // The number determines how good our anti-aliasing works.
        // Possible values are 2,4,8,16,32, but not all work on all computers.
        // 4 is safe, and 8 is too in almost all cases
        // Higher numbers mean lower frame-rates
        _player.BeginRun();
        _player.Initialize(m_UFO, m_Pod);
        _player.GameOver = true;
        _player.Active = false;
        m_UFOTimer.Amount = m_UFOTimerSeedAmount;
        m_UFO.Initialize(_player);
        m_Pod.Initialize(_player, m_UFO);
        m_Pod.BeginRun();
        m_PodTimer.Amount = m_PodTimerSeedAmount;
        m_PlayerClear.Radius = 150;
        m_PlayerClear.Moveable = false;
        m_BackGroundPlay.Amount = m_Background.Duration.Seconds;
        m_AtariHUD.ProcessWords("ATARI INC", new Vector3(34, (-Serv.WindowHeight * 0.5f) + 20, 0), 5);
        m_AtariDate.ProcessNumber(1980, new Vector3(-34, (-Serv.WindowHeight * 0.5f) + 20, 0), 5);
    }

    /// <summary>
    /// Allows the game to run logic such as updating the world,
    /// checking for collisions, gathering input, and playing audio.
    /// </summary>
    /// <param name="gameTime">Provides a snapshot of timing values.</param>
    protected override void Update(GameTime gameTime)
    {
        if (_canStep)
        {
            m_KeyState = Keyboard.GetState();
            KeyInput();
            m_KeyStateOld = m_KeyState;

            if (m_Paused)
                return;

            if (_player.CheckClear)
            {
                if (CheckPlayerClear())
                {
                    _player.Spawn = true;
                }
            }
            else if (!_player.GameOver)
            {
                PlayBackground();
            }

            UFOController();
            PodController();
            RockController();

            base.Update(gameTime);

            if (IsAIEnv)
                _canStep = false;
        }
    }

    /// <summary>
    /// This is called when the game should draw itself.
    /// </summary>
    /// <param name="gameTime">Provides a snapshot of timing values.</param>
    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(new Color(new Vector3(0.01666f, 0, 0.1f)));

        base.Draw(gameTime);
    }

    void KeyInput()
    {
#if DEBUG
        if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
            Keyboard.GetState().IsKeyDown(Keys.Escape))
            Exit();

        if (m_KeyState.IsKeyDown(Keys.K) && !m_KeyStateOld.IsKeyDown(Keys.K))
        {
            m_Pod.Reset();
            m_Pod.Spawn();
        }
#endif
        if (m_KeyState.IsKeyDown(Keys.P) && !m_KeyStateOld.IsKeyDown(Keys.P))
        {
            m_Paused = !m_Paused;
        }

        if (m_KeyState.IsKeyDown(Keys.N) && !m_KeyStateOld.IsKeyDown(Keys.N) && _player.GameOver &&
            !_player.NewHighScore)
        {
            _player.GameOver = false;
            NewGame();
        }
    }

    void PlayBackground()
    {
        if (m_BackgroundDelay.Expired)
        {
            m_BackgroundDelay.Reset();

            if (m_BackgroundDelay.Amount > 0.25f)
                m_BackgroundDelay.Amount -= 0.025f;


            if (m_BackGroundPlay.Expired && m_PlayedBack)
            {
                m_BackGroundPlay.Reset();
                m_Background.Play(0.25f, 0, 0);
            }
        }
    }

    bool CheckPlayerClear()
    {
        foreach (Rock rock in m_Rocks)
        {
            if (rock.Active)
            {
                if (m_PlayerClear.CirclesIntersect(rock.Position, rock.Radius))
                    return false;
            }
        }

        if (m_UFO.Active)
        {
            if (m_PlayerClear.CirclesIntersect(m_UFO.Position, m_UFO.Radius))
                return false;
        }

        if (m_UFO.Shot.Active)
            return false;

        if (m_Pod.Active)
        {
            if (m_PlayerClear.CirclesIntersect(m_Pod.Position, m_Pod.Radius))
                return false;
        }
        else
        {
            for (int pair = 0; pair < 3; pair++)
            {
                if (m_Pod.PodPair[pair].Active)
                {
                    if (m_PlayerClear.CirclesIntersect(m_Pod.PodPair[pair].Position, m_Pod.PodPair[pair].Radius))
                        return false;
                }
                else
                {
                    for (int pod = 0; pod < 2; pod++)
                    {
                        if (m_Pod.PodPair[pair].Pods[pod].Active)
                        {
                            if (m_PlayerClear.CirclesIntersect(m_Pod.PodPair[pair].Pods[pod].Position,
                                m_Pod.PodPair[pair].Pods[pod].Radius))
                                return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    void NewGame()
    {
        _player.NewGame();
        ResetUFO();
        ResetPod();

        foreach (Rock rock in m_Rocks)
        {
            rock.Active = false;
        }

        m_Wave = 0;
        m_UFOCount = 0;
        m_LargeRockSpawnAmount = 2;
        m_BackgroundDelay.Amount = m_BackgroundOneDelaySeed;
        m_BackgroundTwoDelay.Amount = m_BackgroundTwoDelaySeed;

    }

    void UFOController()
    {
        if (m_UFO.Done || m_UFO.Hit)
        {
            ResetUFO();
        }

        if (m_UFOTimer.Expired && !m_UFO.Active)
        {
            m_UFOTimer.Amount = Serv.RandomMinMax(m_UFOTimerSeedAmount * 0.5f,
                m_UFOTimerSeedAmount + (m_UFOTimerSeedAmount - m_Wave));
            m_UFO.Spawn(m_UFOCount, m_Wave);
            m_UFOCount++;
        }
    }

    void PodController()
    {
        if (m_RockCount < 4 && m_Wave > 1)
        {
            if (!m_Pod.Done)
                m_PodTimer.Reset();

            if (m_PodTimer.Expired)
            {
                m_PodTimer.Reset();

                if (m_Pod.Done)
                {
                    m_Pod.Spawn();
                }
            }
        }
        else
            m_PodTimer.Reset();
    }

    void ResetUFO()
    {
        m_UFOTimer.Reset();
        m_UFO.Active = false;
        m_UFO.Done = false;
        m_UFO.Hit = false;
    }

    void ResetPod()
    {
        m_PodTimer.Reset();
        m_Pod.Reset();
    }

    void RockController()
    {
        m_RockCount = 0;

        foreach (Rock rock in m_Rocks)
        {
            if (rock.Active)
            {
                m_RockCount++;

                if (rock.Hit)
                {
                    switch (rock.SizeofRock)
                    {
                        case RockSize.Large:
                            SpawnRocks(rock.Position, RockSize.Medium, 2);
                            break;

                        case RockSize.Medium:
                            SpawnRocks(rock.Position, RockSize.Small, 2);
                            break;

                        case RockSize.Small:
                            break;
                    }

                    rock.Active = false;
                    rock.Hit = false;
                    return;
                }
            }
        }

        if (m_RockCount == 0)
        {
            if (m_LargeRockSpawnAmount > 12)
                m_LargeRockSpawnAmount = 12;

            SpawnRocks(Vector3.Zero, RockSize.Large, m_LargeRockSpawnAmount += 2);
            m_Wave++;
        }
    }

    void SpawnRocks(Vector3 position, RockSize rockSize, int count)
    {
        for (int i = 0; i < count; i++)
        {
            bool spawnNewRock = true;
            int rockFree = m_Rocks.Count;

            for (int rock = 0; rock < rockFree; rock++)
            {
                if (!m_Rocks[rock].Active && !m_Rocks[rock].ExplosionActive)
                {
                    spawnNewRock = false;
                    rockFree = rock;
                    break;
                }
            }

            if (spawnNewRock)
            {
                m_Rocks.Add(new Rock(this));
                m_Rocks.Last().Create(_player, m_UFO);
                m_Rocks.Last().LoadSound(m_RockExplode);
            }

            m_Rocks[rockFree].Spawn(position, rockSize);
        }
    }


    private TimeSpan _totalGameTime;
    private int _prevScore = 0;
    public Step Step(int act)
    {
        _player.Step(act);

        _canStep = true;

        TimeSpan elapsedTime = TargetElapsedTime;
        GameTime gameTime = new GameTime(_totalGameTime, elapsedTime);
        _totalGameTime += elapsedTime;

        _player.Step(act);

        this.RunOneFrame();

        System.Threading.Thread.Sleep(10);

        Step step = new Step();
        step.Done = _player.GameOver;
        step.Reward = _player.Score - _prevScore;
        _prevScore = _player.Score;

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
        var crop = Resize(Serv.RenderTarget, new Rectangle(0, 0, desiredWidth, desiredHeight));

        backBuffer = new int[desiredWidth * desiredHeight];
        crop.GetData(backBuffer);

        float[,,] b = new float[desiredWidth, desiredHeight, 3];
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

        // ------- For testing purpose --------
        //buffer = new byte[desiredWidth * desiredHeight * 3];
        //int bIndex = 0;
        //for (int i = 0; i < backBuffer.Length; i++)
        //{
        //    var color = new Color((uint)backBuffer[i]);
        //    buffer[bIndex] = color.B;
        //    buffer[bIndex + 1] = color.G;
        //    buffer[bIndex + 2] = color.R;

        //    bIndex += 3;
        //}

        //var bmp = new System.Drawing.Bitmap(desiredWidth, desiredHeight);
        //var data = bmp.LockBits(new System.Drawing.Rectangle(0, 0, desiredWidth, desiredHeight), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        //System.Runtime.InteropServices.Marshal.Copy(buffer, 0, data.Scan0, buffer.Length);
        //bmp.UnlockBits(data);

        //bmp.Save(System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "asteroids.bmp"));

        return b;

    }

    public static Texture2D Resize(Texture2D image, Rectangle destRect)
    {
        Rectangle sourceRect = new Rectangle(0, 0, image.Width, image.Height);
        var graphics = image.GraphicsDevice;
        var ret = new RenderTarget2D(graphics, destRect.Width, destRect.Height);
        var sb = new SpriteBatch(graphics);

        graphics.SetRenderTarget(ret); // draw to image
        graphics.Clear(new Color(0, 0, 0, 0));

        sb.Begin();
        sb.Draw(image, destRect, sourceRect, Color.White);
        sb.End();

        graphics.SetRenderTarget(null); // set back to main window

        return (Texture2D)ret;
    }

    public NDArray Reset()
    {
        _player.GameOver = false;
        _prevScore = 0;
        NewGame();

        System.Threading.Thread.Sleep(100);
        Step step = new Step();
        step.Observation = GetScreenBuffer();

        return step.Observation;
    }

    public void CloseEnvironment()
    {
        Exit();
    }
}