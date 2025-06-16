using DeepSharp.RL.Environs;
using TorchSharp.Modules;

namespace EfficientZero;

public class ReplayBuffer
{
    private readonly Config _config;
    private readonly int _size; // How many game records to store
    private List<GameRecord> _buffer; // List of stored game records
    private int _totalVals; // How many total steps are stored
    private int _gameCount; //total number of saved games

    private readonly bool _prioritizedReplay;
    private readonly float _priorityAlpha;
    private readonly float _priorityBeta;

    // List of start points of each game if the whole buffer were concatenated
    private List<int> _gameStartsList;

    private readonly int _rewardDepth;
    private readonly int _rolloutDepth;
    private List<float> _priorities;
    private List<float> _batchPriorities;


    public ReplayBuffer(Config config)
    {
        _config = config;
        _size = config.buffer_size;
        _buffer = new List<GameRecord>();
        _totalVals = 0;
        _gameCount = 0;

        _prioritizedReplay = config.priority_replay;
        _priorityAlpha = config.priority_alpha;
        _priorityBeta = config.priority_beta;

        _gameStartsList = new List<int>();
        _rewardDepth = config.reward_depth;
        _rolloutDepth = config.rollout_depth;
        _priorities = new List<float>();
        _batchPriorities = new List<float>();
    }

    public void UpdateStats()
    {
        // Maintain stats for the total length of all games in the buffer
        // and where each game would begin if all games were concatenated
        // so that each step of each game can be uniquely indexed

        List<int> lengths = _buffer.Select(x => x.Values.Count).ToList();
        _gameStartsList = Enumerable.Range(0, _buffer.Count)
                                   .Select(i => lengths.Take(i).Sum())
                                   .ToList();
        _totalVals = lengths.Sum();
        _priorities = _buffer.SelectMany(x => x.Priorities).ToList();
        _priorities = _priorities.Select(p => (float)Math.Pow(p, _priorityAlpha)).ToList();
        float sumPriorities = _priorities.Sum();
        _priorities = _priorities.Select(p => p / sumPriorities).ToList();
    }

    public void SaveGame(GameRecord game)
    {
        // If reached the max size, remove the oldest GameRecord, and update stats accordingly
        if (_buffer.Count >= _size)
        {
            _buffer.RemoveAt(0);
        }

        _buffer.Add(game);
        var prioritySum = game.Priorities.Sum();
        _batchPriorities.Add(prioritySum);

        UpdateStats();
        _gameCount += 1;
    }

    public List<BufferData> GetBatch(int batchSize = 40)
    {
        List<BufferData> batch = new List<BufferData>();

        // Get a random list of points across the length of the buffer to take training examples
        float[] probabilities = _prioritizedReplay ? _priorities.ToArray() : null;
        int[] startVals = Utils.GetIndices(_totalVals, batchSize, probabilities);
        List<float> weightList = new List<float>();
        List<float> prevWeights = new List<float>();
        foreach (int val in startVals)
        {
            // Get the index of the game in the buffer (buf_ndx) and a location in the game (game_ndx)
            (int bufNdex, int gameNdex) = GetNdxs(val);

            GameRecord game = _buffer[bufNdex];

            // Gets a series of actions, values, rewards, policies, up to a depth of rollout_depth
            var buffer = game.MakeTarget(gameNdex, _rewardDepth, _rolloutDepth + 1);

            float weight = 1 / _priorities[val];
            buffer.Weight = weight;
            prevWeights.Add(weight);
            float weight2 = (float)Math.Pow(_priorities.Count * _priorities[val], -_priorityBeta);
            weightList.Add(weight2);

            batch.Add(buffer);
        }

        float weightMax = weightList.Max();
        for (int i = 0; i < batch.Count; i++)
        {
            weightList[i] = weightList[i] / weightMax;
            weightList[i] = Math.Clamp(weightList[i], .01f, 1f);

            batch[i].Weight = weightList[i];
        }

        return batch;
    }

    public (int bufNdex, int gameNdex) GetNdxs(int val)
    {
        if (val >= _totalVals)
        {
            throw new ArgumentOutOfRangeException("Trying to get a value beyond the length of the buffer");
        }

        // Assumes len_list is sorted, gets the last entry in starts_list which is below val
        // by iterating through game_starts_list until one is above val, at which point
        // it returns the previous value in game_starts_list
        // and the position in the game is gap between the game's start position and val
        for (int i = 0; i < _gameStartsList.Count; i++)
        {
            if (_gameStartsList[i] > val)
            {
                return (i - 1, val - _gameStartsList[i - 1]);
            }
        }
        return (_buffer.Count - 1, val - _gameStartsList[^1]);
    }

    public void Reanalyse(MCTS mcts)
    {
        for (int i = 0; i < _buffer.Count; i++)
        {
            // Reanalyse on average every 50 games at max size
            double chance = new Random().NextDouble();
            if (chance < 2.0 / _buffer.Count)
            {
                GameRecord newGame = _buffer[i].Reanalyse(mcts);
                _buffer[i] = newGame;
            }
        }
    }
}

public class BufferData
{
    public BufferData(List<Observation> preObs, List<Act> actions, List<float> targetValues, List<float> targetRewards, List<List<float>> targetPolicies)
    {
        PreObs = preObs;
        Actions = actions;
        TargetValues = targetValues;
        TargetRewards = targetRewards;
        TargetPolicies = targetPolicies;
    }

    public List<Observation> PreObs { get; set; }
    public List<Act> Actions { get; set; }
    public List<float> TargetValues { get; set; }
    public List<float> TargetRewards { get; set; }
    public List<List<float>> TargetPolicies { get; set; }
    public float Weight { get; set; }
}