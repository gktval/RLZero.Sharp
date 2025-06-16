using nkast.Aether.Physics2D.Dynamics.Contacts;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TopDownCarPhysics.Entities;

namespace TopDownCarPhysics;

public class FrictionDetector
{
    private Environment _env { get; set; }

    public FrictionDetector(Environment env)
    {
        _env = env;
    }

    private void DoContact(Contact contact, bool begin)
    {
        RoadTile tile = contact.FixtureA.Body.Tag as RoadTile;
        Vehicle car = contact.FixtureB.Body.Tag as Vehicle;
        if (tile == null)
        {
            tile = contact.FixtureB.Body.Tag as RoadTile;
            car = contact.FixtureA.Body.Tag as Vehicle;
        }
        if (tile == null || car == null)
        {
            //this is the edge of the world
            if (tile == null && car != null)
            {
                _env.Reward += -100;
                _env.IsDone = true;
            }
            return;
        }
        if (begin)
        {
            tile.Color = _env.RoadColor;
            car.Tiles.Add(tile);
            if (!tile.RoadVisited)
            {
                tile.RoadVisited = true;
                _env.Reward += 1000f / (float)_env.Track.Length;
                _env.TileVisitedCount++;
                float visitpct = (float)_env.TileVisitedCount / (float)_env.Track.Length;
                if (tile.Index == 0 && visitpct > _env.LapCompletePercent)
                {
                    _env.NewLap = true;
                }
            }
        }
        else
        {
            car.Tiles.Remove(tile);
        }
    }
    public bool BeginContact(Contact contact)
    {
        DoContact(contact, true);
        return (false);
    }

    public void EndContact(Contact contact)
    {
        DoContact(contact, false);
    }

}
