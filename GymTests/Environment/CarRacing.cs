using DeepSharp.RL.Environs;
using static TorchSharp.torch;
using TorchSharp;
using DeepSharp.RL.Environs.Spaces;
using System;
using static TorchSharp.torch.distributions.constraints;
using NumSharp;
using Microsoft.Xna.Framework;

namespace LunarLander;

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

    public CarRacing(DeviceType deviceType = DeviceType.CUDA, int maxEpisodes = 500, bool isImage = true)
        : base("CarRacing", deviceType)
    {
        Initialise();

        _maxEpisodes = maxEpisodes;
        _isImage = isImage;

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

    public  void Initialise()
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
        int actVal = act.Value.item<int>();
        var (observation, reward, _done, information) = GymEnv.Step(actVal);
        _storedActions.Add(act);

        GymEnv.Render = Render;
        GymEnv.RunOneFrame();

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
        IsDone = _done;
        _reward = reward;
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

    public override Environ<Space, Space> Clone()
    {
        var env = new CarRacing(Device.type, _maxEpisodes);
        env.ActionSpace = new Disperse(_actSize, torch.ScalarType.Int32, Device.type);
        env.ObservationSpace = new Disperse(_obsSize, torch.ScalarType.Float32, Device.type);
        env.Reward = new Reward(Reward.Value);
        env.IsDone = IsDone;

        for (int i = 0; i < _storedActions.Count; i++)
        {
            Act act = _storedActions[i];
            env.Step(act, i);
        }

        return env;
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
