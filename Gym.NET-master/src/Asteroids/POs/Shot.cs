﻿using Microsoft.Xna.Framework;
using System.Drawing;

namespace Asteroids_Deluxe
{
    public class Shot : VectorEngine.Vector
    {
        VectorEngine.Timer m_LifeTimer;

        public Shot(AsteroidGame game) : base(game)
        {
            m_LifeTimer = new VectorEngine.Timer(game);
        }

        public override void Initialize()
        {
            base.Initialize();
        }

        public override void Update(GameTime gameTime)
        {
            base.Update(gameTime);
            CheckBorders();

            if (m_LifeTimer.Expired)
            {
                Active = false;
            }
        }

        public void Spawn(Vector3 position, Vector3 velecity, float timer)
        {
            m_LifeTimer.Amount = timer;
            Position = position;
            Velocity = velecity;
            Active = true;
        }

        public void Reset()
        {
            Active = false;
        }

        protected override void InitializeLineMesh()
        {
            Vector3[] pointPosition = new Vector3[5];
            float size = 3f;

            pointPosition[0] = new Vector3(-size, -size, 0);
            pointPosition[1] = new Vector3(size, -size, 0);
            pointPosition[2] = new Vector3(size, size, 0);
            pointPosition[3] = new Vector3(-size, size, 0);
            pointPosition[3] = new Vector3(-size, -size, 0);

            Radius = InitializePoints(pointPosition);

            Active = false;
        }
    }
}
