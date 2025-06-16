using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TopDownCarPhysics.Utils;

public delegate Vector2 PositionDelegate();

public class GameCamera : OrthographicCamera
{
    private bool _shaking;
    private float _shakingMagnitude;
    private TimeSpan _shakingTimer;
    private Vector2 _initialPosition;

    private float _initialZoom;
    private bool _targetingZoom;
    private float _targetZoom;
    //private CountdownTimer _targetZoomTimer;

    //private PositionDelegate _followPositionDelegate;
    private bool _followingPosition;
    private Random _random;
    private bool _clampToMap;
    private Vector2 _followPosition;

    private Vector2 _mapSize;
    private int _tileSize;

    public GameCamera(ViewportAdapter viewportAdapter, int mapWidth, int mapHeight, int tileSize) : base(viewportAdapter)
    {
        _targetingZoom = false;
        _targetZoom = 1f;
        _random = new Random();

        //_followPositionDelegate = null;
        _followingPosition = false;
        _clampToMap = true;

        MaximumZoom = 5f;
        MinimumZoom = 0.01f;
        Origin = Vector2.Zero;

        _mapSize.X = mapWidth;
        _mapSize.Y = mapHeight;
        _tileSize = tileSize;
    }

    public void SetViewPortAdapter(ViewportAdapter viewportAdapter)
    {
        ResetViewPortAdapter(viewportAdapter);
    }

    public Rectangle MapRectangle
    {
        get
        {
            return new Rectangle(new Point((int)Position.X, (int)Position.Y), new Point((int)ViewportAdapter.VirtualWidth / 2, (int)ViewportAdapter.VirtualHeight / 2));
        }
    }

    public void Update(GameTime gameTime, Vector2 position)
    {

        if (_shaking)
        {
            if (_shakingTimer.TotalMilliseconds > 0)
            {
                _shakingTimer -= gameTime.ElapsedGameTime;
                Position = _initialPosition + new Vector2((float)(_random.NextDouble() * _shakingMagnitude));
            }
            else
            {
                _shaking = false;
                _shakingTimer = TimeSpan.Zero;
                Position = _initialPosition;
            }
        }

        Vector2 lookAt = position * _tileSize;
        lookAt += new Vector2(_tileSize / 2, _tileSize / 2);
        if (_clampToMap)
        {
            float scale = 1f * ViewportScale;
            var cameraMin = new Vector2(BoundingRectangle.Width * Zoom / scale, BoundingRectangle.Height * Zoom / scale);
            //lookAt += cameraMin;
            //lookAt = MapClampedPosition(lookAt);
        }

        LookAt(lookAt);

        //if (_targetingZoom)
        //{
        //    _targetZoomTimer.Update(gameTime);

        //    var delta = _targetZoomTimer.CurrentTime.TotalSeconds / _targetZoomTimer.Interval.TotalSeconds;
        //    var currentZoom = MathHelper.Lerp(_initialZoom, _targetZoom, (float)delta);
        //    Zoom = currentZoom;

        //    if (Math.Abs(Zoom - _targetZoom) < 0.0001f)
        //    {
        //        Zoom = _targetZoom;
        //        _targetingZoom = false;
        //    }
        //}
    }

    private Vector2 MapClampedPosition(Vector2 position)
    {
        float scale = 2f * ViewportScale;
        //position = new Vector2(position.X + ViewportAdapter.VirtualWidth / 2, position.Y + ViewportAdapter.VirtualHeight / 2);
        var cameraMax = new Vector2(_mapSize.X * _tileSize - (BoundingRectangle.Width * Zoom / scale),
            _mapSize.Y * _tileSize - (BoundingRectangle.Height * Zoom / scale));
        var cameraMin = new Vector2(BoundingRectangle.Width * Zoom / scale, BoundingRectangle.Height * Zoom / scale);
        return Vector2.Clamp(position, cameraMin, cameraMax);
    }

    // Time in seconds
    public void Shake(float time, float magnitude)
    {
        //SoundManager.PlaySound(Assets.GetSound("Audio/SE/shake"));

        if (_shaking)
            return;

        _shaking = true;
        _shakingMagnitude = magnitude;
        _shakingTimer = TimeSpan.FromSeconds(time);
        _initialPosition = Position;
    }

    public void ZoomTo(float targetZoom, double time, Vector2? origin = null)
    {
        if (_targetingZoom)
            return;

        _initialZoom = Zoom;
        _targetZoom = targetZoom;
        _targetingZoom = true;
        //_targetZoomTimer = new CountdownTimer(time);
        //_targetZoomTimer.Start();

        if (origin.HasValue)
            Origin = origin.Value;
    }

    public void FollowPosition(Vector2 position)
    {
        _followPosition = position;
        //_followPositionDelegate = new PositionDelegate(getPosition);
        _followingPosition = true;
    }

    public void RemoveFollowPosition()
    {
        _followPosition = Vector2.Zero;
        //_followPositionDelegate = null;
        _followingPosition = false;
    }
}
