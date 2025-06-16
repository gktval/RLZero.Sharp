using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace EfficientZero;

public class Buffer
{
    private Dictionary<string, object> config;
    private Memory memory;
    private int imageSize;
    private DateTime lastTime; // Used if profiling speed of batching
    private int size; // How many game records to store
    private float priorityAlpha;
    private float tau;

    // List of start points of each game if the whole buffer were concatenated
    private List<int> gameStartsList;

    private List<object> buffer;
    private List<int> bufferIndices;
    private bool prioritizedReplay;
    private List<float> priorities;
    private int totalVals;

    public Buffer(Dictionary<string, object> config, Memory memory)
    {
        this.config = config;
        this.memory = memory;
        this.imageSize = (int)config["full_image_size"];

        this.lastTime = DateTime.Now; // Used if profiling speed of batching

        this.size = (int)config["buffer_size"]; // How many game records to store
        this.priorityAlpha = (float)config["priority_alpha"];
        this.tau = (float)config["tau"];

        this.gameStartsList = new List<int>();

        if ((bool)config["load_buffer"] && File.Exists(Path.Combine("buffers", (string)config["env_name"])))
        {
            LoadBuffer();
        }
        else
        {
            this.buffer = new List<object>();
            this.bufferIndices = new List<int>();
        }

        this.prioritizedReplay = (bool)config["priority_replay"];
        this.priorities = new List<float>();
    }

    public void SaveBuffer()
    {
        using (FileStream fs = new FileStream(Path.Combine("buffers", (string)config["env_name"]), FileMode.Create))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(fs, new Tuple<List<object>, List<int>>(this.buffer, this.bufferIndices));
        }
    }

    public void LoadBuffer()
    {
        using (FileStream fs = new FileStream(Path.Combine("buffers", (string)config["env_name"]), FileMode.Open))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            var data = (Tuple<List<object>, List<int>>)formatter.Deserialize(fs);
            this.buffer = data.Item1;
            this.bufferIndices = data.Item2;
        }
        UpdateStats();
    }

    public void UpdateValues(int index, object values)
    {
        try
        {
            int bufferIndex = bufferIndices.IndexOf(index);
            ((dynamic)buffer[bufferIndex]).values = values;
            int totalGames = (int)((dynamic)memory.GetType().GetMethod("get_total_games").Invoke(memory, null));
            ((dynamic)buffer[bufferIndex]).last_analysed = totalGames;
        }
        catch (ArgumentOutOfRangeException)
        {
            Console.WriteLine($"No buffer item with index {index}");
        }
    }

    public void AddPriorities(int index, bool reanalysing = false)
    {
        try
        {
            int bufferIndex = bufferIndices.IndexOf(index);
            ((dynamic)buffer[bufferIndex]).AddPriorities((int)config["reward_depth"], reanalysing);
        }
        catch (ArgumentOutOfRangeException)
        {
            Console.WriteLine($"No buffer item with index {index}");
        }
    }

    public void UpdateStats()
    {
        // Maintain stats for the total length of all games in the buffer
        // and where each game would begin if all games were concatenated
        // so that each step of each game can be uniquely indexed

        List<int> lengths = buffer.Select(x => ((dynamic)x).values.Length).ToList();
        gameStartsList = Enumerable.Range(0, buffer.Count).Select(i => lengths.Take(i).Sum()).ToList();
        int totalValues = lengths.Sum();
        priorities = buffer.SelectMany(x => ((dynamic)x).priorities).ToList();
        priorities = priorities.Select(p => (float)Math.Pow((double)p, (double)priorityAlpha)).ToList();
        float sumPriorities = priorities.Sum();
        priorities = priorities.Select(p => p / sumPriorities).ToList();
    }

    public (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) GetBatch(int batchSize = 40, Device device)
    {
        PrintTiming("start");
        int rolloutDepth = (int)config["rollout_depth"];
        List<object> batch = new List<object>();

        // Get a random list of points across the length of the buffer to take training examples
        float[] probabilities = prioritizedReplay ? priorities.ToArray() : null;

        if (probabilities != null && probabilities.Length != totalVals)
        {
            // Breakpoint equivalent can be implemented if needed
        }
        int[] startValues = Enumerable.Range(0, totalVals).OrderBy(x => Guid.NewGuid()).Take(batchSize).ToArray();
        PrintTiming("get indices");

        float[,,] imagesArray = new float[batchSize, rolloutDepth, imageSize];
        long[,] actionsArray = new long[batchSize, rolloutDepth];
        float[,] targetValuesArray = new float[batchSize, rolloutDepth];
        float[,] targetRewardsArray = new float[batchSize, rolloutDepth];
        float[,,] targetPoliciesArray = new float[batchSize, rolloutDepth, (int)config["action_size"]];
        float[] weightsArray = new float[batchSize];
        long[] depthsArray = new long[batchSize];

        for (int i = 0; i < startValues.Length; i++)
        {
            // Get the index of the game in the buffer (gameIndex) and a location in the game (frameIndex)
            (int gameIndex, int frameIndex) = GetIndices(startValues[i]);

            var game = buffer[gameIndex];

            int rewardDepth = GetRewardDepth(startValues[i], tau, (int)config["total_training_steps"], (int)config["reward_depth"]);

            // Gets a series of actions, values, rewards, policies, up to a depth of rolloutDepth
            (object images, object actions, object targetValues, object targetRewards, object targetPolicies, int depth) =
                ((dynamic)game).MakeTarget(frameIndex, (int)config["reward_depth"], (int)config["rollout_depth"]);

            // Add tuple to batch
            float weight = prioritizedReplay ? 1 / priorities[startValues[i]] : 1;

            imagesArray[i] = (float[,])images;
            actionsArray[i] = (long[,])actions;
            targetValuesArray[i] = (float[,])targetValues;
            targetRewardsArray[i] = (float[,])targetRewards;
            targetPoliciesArray[i] = (float[,,])targetPolicies;

            weightsArray[i] = weight;
            depthsArray[i] = depth;
        }
        PrintTiming("make_lists");

        Tensor imagesTensor = torch.from_array(imagesArray, device);
        Tensor actionsTensor = torch.from_array(actionsArray, device);
        Tensor targetValuesTensor = torch.from_array(targetValuesArray, device);
        Tensor targetPoliciesTensor = torch.from_array(targetPoliciesArray, device);
        Tensor targetRewardsTensor = torch.from_array(targetRewardsArray, device);
        Tensor weightsTensor = torch.from_array(weightsArray, device);
        weightsTensor = weightsTensor / weightsTensor.max();
        PrintTiming("make_tensors");
        return (imagesTensor, actionsTensor, targetValuesTensor, targetRewardsTensor, targetPoliciesTensor, weightsTensor, depthsArray);
    }

    public object GetBufferIndex(int index)
    {
        int bufferIndex = bufferIndices.IndexOf(index);
        return buffer[bufferIndex];
    }

    public int GetBufferLength()
    {
        return buffer.Count;
    }

    public List<object> GetBuffer()
    {
        return buffer;
    }

    public int[] GetBufferIndices()
    {
        return bufferIndices.ToArray();
    }

    public float[] GetReanalyseProbabilities()
    {
        int totalGames = (int)((dynamic)memory.GetType().GetMethod("get_total_games").Invoke(memory, null));
        float[] probabilities = buffer.Select(x => (float)(totalGames - ((dynamic)x).last_analysed)).ToArray();
        if (probabilities.Sum() > 0)
        {
            return probabilities.Select(p => p / probabilities.Sum()).ToArray();
        }
        else
        {
            return new float[0];
        }
    }

    public void SaveGame(object game, int nFrames, float score, Dictionary<string, int> gameData)
    {
        // If reached the max size, remove the oldest GameRecord, and update stats accordingly
        while (this.buffer.Count >= this.size)
        {
            this.buffer.RemoveAt(0);
            this.bufferIndices.RemoveAt(0);
        }

        this.buffer.Add(game);
        UpdateStats();
        SaveBuffer();
        this.bufferIndices.Add(gameData["games"] - 1);
    }

    public (int, int) GetIndices(int val)
    {
        if (val >= this.buffer.Count)
        {
            throw new ArgumentOutOfRangeException("Trying to get a value beyond the length of the buffer");
        }

        // Assumes gameStartsList is sorted, gets the last entry in startsList which is below val
        // by iterating through gameStartsList until one is above val, at which point
        // it returns the previous value in gameStartsList
        // and the position in the game is gap between the game's start position and val
        for (int i = 0; i < this.gameStartsList.Count; i++)
        {
            if (this.gameStartsList[i] > val)
            {
                return (i - 1, val - this.gameStartsList[i - 1]);
            }
        }
        return (this.buffer.Count - 1, val - this.gameStartsList[this.gameStartsList.Count - 1]);
    }

    public int GetRewardDepth(int val, float tau = 0.3f, int totalSteps = 100000, int maxDepth = 5)
    {
        int depth;

        if ((bool)this.config["off_policy_correction"])
        {
            // Varying reward depth depending on the length of time since the trajectory was generated
            // Follows the formula in A.4 of EfficientZero paper
            int stepsAgo = this.buffer.Count - val;
            depth = maxDepth - (int)Math.Floor((double)(stepsAgo / (tau * totalSteps)));
            depth = Math.Clamp(depth, 1, maxDepth);
        }
        else
        {
            depth = maxDepth;
        }
        return depth;
    }

    public void PrintTiming(string tag, double minTime = 0.05)
    {
        if ((bool)this.config["get_batch_profiling"])
        {
            DateTime now = DateTime.Now;
            Console.WriteLine($"{tag,-20} {now - this.lastTime}");
            this.lastTime = now;
        }
    }
}
