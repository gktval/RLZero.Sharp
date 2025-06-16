using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;

using DeepSharp.RL.Environs;
using static TorchSharp.torch;
using TorchSharp;
using DeepSharp.RL.Environs.Spaces;
using System;
using static TorchSharp.torch.distributions.constraints;

namespace LunarLander;

public class CartPole : Environ<Space, Space>
{
    public int _stepCounter;
    public int _maxSteps;
    public int _obsSize;
    private int _actSize;
    private float _reward;
    private int _maxEpisodes;
    private List<Act> _storedActions;
    public CartPoleEnv GymEnv { get; set; }
    public bool IsDone { get; set; }

    public CartPole(DeviceType deviceType = DeviceType.CUDA, int maxEpisodes = 500)
        : base("Cartpole", deviceType)
    {
        Initialise();

        _maxEpisodes = maxEpisodes;

        _actSize = GymEnv.ActionSpace.Shape.Size;
        _obsSize = GymEnv.ObservationSpace.Shape.Size;
        _storedActions = new List<Act>();

        ActionSpace = new Disperse(_actSize, torch.ScalarType.Int32, deviceType);
        ObservationSpace = new Disperse(_obsSize, torch.ScalarType.Float32, deviceType);
        Observation = Reset();

        CallBack = OnStepCallback;
    }

    public void Initialise()
    {
        GymEnv = new CartPoleEnv(WinFormEnvViewer.Factory);
        GymEnv.Seed(42);
        GymEnv.Reset();
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
        Observation =  new Observation(torch.tensor(obs.ToFloatArray()).to(Device));
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

        if (Render)
            GymEnv.Render();

        var myState = observation.ToFloatArray();
        IsDone = _done;
        _reward = reward;
        Reward = new Reward(Reward.Value + _reward);

        if (_stepCounter > _maxSteps)
            IsDone = true;

        return new Observation(torch.tensor(myState).to(Device));
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
        var env = new CartPole(Device.type, _maxEpisodes);
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
