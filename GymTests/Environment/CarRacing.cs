using DeepSharp.RL.Environs;
using static TorchSharp.torch;
using TorchSharp;
using DeepSharp.RL.Environs.Spaces;
using System;
using static TorchSharp.torch.distributions.constraints;
using NumSharp;
using Microsoft.Xna.Framework;

namespace GymTest;

public class CarRacing : Environ<Space, Space>
{
    public int _stepCounter;
    public int _maxSteps;
    public int _obsSize;
    private int _actSize;
    private float _reward;
    private int _maxEpisodes;
    private List<Act> _storedActions;
    private bool _isImage;
    public TopDownCarPhysics.GameMain GymEnv { get; set; }
    public bool IsDone { get; set; }
    /// <summary>
    /// The number of frames to skip.
    /// a 1 means no skipping. 2 means every other
    /// </summary>
    public int FrameSkip = 1;

    public CarRacing(DeviceType deviceType = DeviceType.CUDA, int maxEpisodes = 500, bool isImage = true, int frameSkip = 1)
        : base("CarRacing", deviceType)
    {
        Initialise();

        _maxEpisodes = maxEpisodes;
        _isImage = isImage;
        FrameSkip = frameSkip;

        _actSize = 5; // GymEnv.ActionSpace.Shape.Size;
        if (_isImage)
            _obsSize = 96 * 96 * 3;// GymEnv.ObservationSpace.Shape.Size;
        else
            _obsSize = 4;
        _storedActions = new List<Act>();

        ActionSpace = new Disperse(_actSize, torch.ScalarType.Int32, deviceType);
        ObservationSpace = new Disperse(_obsSize, torch.ScalarType.Float32, deviceType);
        ObservationDim = [96, 96, 3];// GymEnv.ObservationSpace.Shape.Dimensions;
        var initState = (NDArray)GymEnv.GetScreenBuffer();
        if (_isImage)
        {
            var myState = initState.ToFloat3DArray();
            Observation = new Observation(torch.from_array(myState).to(Device));
        }
        else
        {
            var myState = initState.ToFloatArray();
            Observation = new Observation(torch.from_array(myState).to(Device));
        }

        CallBack = OnStepCallback;
    }

    public void Initialise()
    {
        //GymEnv = new CarRacingEnv(WinFormEnvViewer.Factory, continuous: false);
        //if (_isImage)
        //    GymEnv.StateOutputFormat = CarRacingEnv.StateFormat.Pixels;
        //else
        //    GymEnv.StateOutputFormat = CarRacingEnv.StateFormat.Telemetry;
        //GymEnv.Seed(42);
        GymEnv = new TopDownCarPhysics.GameMain(true);
        GymEnv.RunOneFrame();
        _stepCounter = 0;
        _maxSteps = 100000;
        IsDone = false;
    }

    public override Observation Reset()
    {
        var obs = GymEnv.Reset();
        IsDone = false;
        _stepCounter = 0;
        _storedActions.Clear();
        Reward = new Reward(0);
        if (_isImage)
        {
            var myState = obs.ToFloat3DArray();
            Observation = new Observation(torch.from_array(myState).to(Device));
        }
        else
        {
            var myState = obs.ToFloatArray();
            Observation = new Observation(torch.from_array(myState).to(Device));
        }
        ObservationList = new List<Observation> { Observation };
        return Observation;
    }

    /// <summary>
    /// Random act
    /// </summary>
    /// <returns></returns>
    public override Act SampleAct()
    {
        int actionsSize = (int)ActionSpace.N;
        var moveProb = torch.randint(0, actionsSize, new torch.Size(1), dtype: ScalarType.Int32).to(Device);
        return new Act(moveProb);
    }

    public override float GetReturn(Episode episode)
    {
        return _reward;
    }

    public override Observation Update(Act act)
    {
        GymEnv.Render = Render;

        int actVal = act.Value.item<int>();
        _reward = 0;

        for (int skip = 0; skip < FrameSkip; skip++)
        {

            bool getObs = skip == FrameSkip - 1;
            var (observation, reward, _done, information) = GymEnv.Step(actVal, getObs);
            _storedActions.Add(act);

            IsDone = _done;
            _reward += reward;

            if (IsDone && !getObs)
            {
                observation = GymEnv.GetScreenBuffer();
                getObs = true;
            }

            if (getObs)
            {
                if (_isImage)
                {
                    var myState = observation.ToFloat3DArray();
                    Observation = new Observation(torch.from_array(myState).to(Device));
                }
                else
                {
                    var myState = observation.ToFloatArray();
                    Observation = new Observation(torch.from_array(myState).to(Device));
                }
            }
            
            if (IsDone)
                break;

        }

        Reward = new Reward(Reward.Value + _reward);

        if (_stepCounter > _maxSteps)
            IsDone = true;

        return Observation;
    }

    public override Reward GetReward(Observation observation)
    {
        return new Reward(_reward);
    }

    public override bool IsComplete(int epoch)
    {
        return IsDone || epoch > _maxEpisodes;
    }

    internal List<Act> GetActions()
    {
        return _storedActions;
    }

    private void OnStepCallback(Step step)
    {

    }

    public override void Close()
    {
        GymEnv.CloseEnvironment();
    }
}
