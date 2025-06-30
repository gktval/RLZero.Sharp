using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using TopDownCarPhysics.Physics;

namespace TopDownCarPhysics.Entities;

internal class Player : Vehicle
{
    private readonly ContentManager _contentManager;
    private SpriteFont _font;
    private Vector2 _initialPosition;
    private float _initialRotation;
    private readonly SpriteBatch _spriteBatch;

    public Vector2 Position => base.Postion;

    public float AngularVelocity => base.AngularVelocity;

    public float AngularDamping => base.AngularDamping;

    public float Rotation => base.Rotation;

    public Vector2 LinearVelocity => base.LinearVelocity;

    public float LinearDamping => base.LinearDamping;

    public Vector2 Forward => base.Forward;

    public Player(Vector2 initialPosition, float initRotation, SpriteBatch spriteBatch, PhysicsWorld physicsWorld, ContentManager contentManager)
        : base(spriteBatch, physicsWorld, contentManager)
    {
        _initialPosition = initialPosition;
        _initialRotation = initRotation;
        _spriteBatch = spriteBatch;
        _contentManager = contentManager;
    }

    public void Reset(Vector2 initialPosition, float initRotation)
    {
        _initialPosition = initialPosition;
        _initialRotation = initRotation;

        InitialisePhysics(_initialPosition, _initialRotation, mass: 1f, turnSpeed: 25f, driftFactor: 0.92f, enableDrifting: true);
    }

    public void LoadContent()
    {
        // Load and initialise the vehicle
        LoadContent("car");

        // Load a font
        _font = _contentManager.Load<SpriteFont>("font");

        // Initialise the vehicle physics, we'll make the player vehicle lighter than
        // the opponent vehicle. If the opponent is heavier, it doesn't deflect much in
        // collisions with the player. However, if we make the opponent lighter (or the
        // player heavier than the opponent) then the opponent car deflects much easier. Try
        // some different values and see what happens ;-)
        //
        // Drift factor: closer to '1' means more 'slippy', start at '0.9' for decent grip
        InitialisePhysics(_initialPosition, _initialRotation, mass: 1f, turnSpeed: 20f, driftFactor: 0.92f, enableDrifting: true);
    }

    public override void Update(GameTime gameTime)
    {
        if (!_isStep)
        {
            // Get keyboard state
            var keyboardState = Keyboard.GetState();


            // Remember to reset the input direction on each update!
            InputDirection = Vector2.Zero;

            // Acceleration and braking
            if (keyboardState.IsKeyDown(Keys.Up)) InputDirection.Y = 1;
            else if (keyboardState.IsKeyDown(Keys.Down)) InputDirection.Y = -1;

            // Turning
            if (keyboardState.IsKeyDown(Keys.Left)) InputDirection.X = -1;
            else if (keyboardState.IsKeyDown(Keys.Right)) InputDirection.X = 1;

            // Press space to skid/drift ;-)
            if (keyboardState.IsKeyDown(Keys.Space)) Skid();

            // Enable/disable drifting/skidding (when disabled it will turn like its on rails)
            if (keyboardState.IsKeyDown(Keys.D)) DisableDrifting();
            if (keyboardState.IsKeyDown(Keys.E)) EnableDrifting();

            // Change the traction/skid control level
            if (keyboardState.IsKeyDown(Keys.I)) ImproveTraction();
            if (keyboardState.IsKeyDown(Keys.R)) ReduceTraction();
        }
        // Process/update this vehicles physics
        base.Update(gameTime);
    }

    private bool _isStep = false;
    public void Step(int action, GameTime gameTime)
    {
        // Remember to reset the input direction on each update!
        InputDirection = Vector2.Zero;

        switch (action)
        {
            case 0:
                //Do Nothing
                break;
            case 1:
                InputDirection.Y = 1; //forward
                break;
            case 2:
                InputDirection.Y = -1; //reverse
                break;
            case 3:
                InputDirection.X = 1; //right
                break;
            case 4:
                InputDirection.X = -1; //left
                break;
            case 5:
                InputDirection.Y = 1; //forward-right
                InputDirection.X = 1;
                break;
            case 6:
                InputDirection.Y = 1; //forward-left
                InputDirection.X = -1;
                break;
            case 7:
                InputDirection.Y = -1; //reverse-right
                InputDirection.X = 1;
                break;
            case 8:
                InputDirection.Y = -1; //reverse-left
                InputDirection.X = -1;
                break;
        }

        _isStep = true;
    }

    public override void Draw()
    {
        // Draw some details about the car physics
        _spriteBatch.DrawString(_font, "Press the arrow keys to turn, brake and accelerate", new Vector2(0, 0), Color.Black);
        _spriteBatch.DrawString(_font, "Press 'I' to improve traction, 'R' to reduce traction", new Vector2(0, 18), Color.Black);
        _spriteBatch.DrawString(_font, "Press/hold space to perform handbrake skid/turn", new Vector2(0, 36), Color.Black);
        _spriteBatch.DrawString(_font, $"Drifting/skidding enabled -> {IsDriftingEnabled} (press 'E' to enable, 'D' to disable)", new Vector2(0, 54), Color.Black);

        base.Draw();
    }
}
