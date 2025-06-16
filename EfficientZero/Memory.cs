using EfficientZero.Mcts;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace EfficientZero;

public class Memory
{
    private readonly Dictionary<string, object> config;
    private readonly string logDirectory;
    private readonly double sessionStartTime;
    private int totalValues;  // How many total steps are stored
    private int totalGames;
    private int totalFrames;
    private int totalBatches;
    private int rewardDepth;
    private int totalTrainingSteps;
    private int rolloutDepth;
    private MinMax minmax;
    private bool isFinished;
    private List<Dictionary<string, object>> gameStats;

    public Memory(Dictionary<string, object> config, string logDirectory)
    {
        this.config = config;
        this.sessionStartTime = DateTime.Now.TimeOfDay.TotalSeconds;
        this.logDirectory = logDirectory;
        this.totalValues = 0;

        var data = new Deserializer().Deserialize<Dictionary<string, object>>(File.ReadAllText(Path.Combine(this.logDirectory, "data.yaml")));
        this.totalGames = Convert.ToInt32(data["games"]);
        this.totalFrames = Convert.ToInt32(data["steps"]);
        this.totalBatches = Convert.ToInt32(data["batches"]);

        this.rewardDepth = Convert.ToInt32(config["reward_depth"]);
        this.totalTrainingSteps = Convert.ToInt32(config["total_training_steps"]);
        this.rolloutDepth = Convert.ToInt32(config["rollout_depth"]);
        this.minmax = new MinMax();
        this.isFinished = false;
        this.gameStats = new List<Dictionary<string, object>>();
    }

    public Dictionary<string, int> GetData()
    {
        return new Dictionary<string, int>
        {
            { "games", this.totalGames },
            { "frames", this.totalFrames },
            { "batches", this.totalBatches }
        };
    }

    public MinMax GetMinMax()
    {
        return this.minmax;
    }

    public void SaveModel(Model model, string logDirectory)
    {
        string path = Path.Combine(logDirectory, "latest_model_dict.pt");
        model.SaveState(path);
    }

    public Model LoadModel(string logDirectory, Model model)
    {
        string path = Path.Combine(logDirectory, "latest_model_dict.pt");
        if (File.Exists(path))
        {
            model.LoadState(path);
        }
        else
        {
            Console.WriteLine($"No dict to load at {path}");
        }

        return model;
    }

    public Dictionary<string, int> DoneGame(int numberOfFrames, int score)
    {
        this.totalGames += 1;
        this.totalFrames += numberOfFrames;
        SaveCoreStats();
        var gameStat = new Dictionary<string, object>
        {
            { "total games", this.totalGames },
            { "score", score },
            { "total frames", this.totalFrames },
            { "elapsed time", GetElapsedTime() },
            { "total batches", this.totalBatches }
        };
        this.gameStats.Add(gameStat);
        if (this.totalGames >= Convert.ToInt32(this.config["max_games"]) || this.totalFrames >= Convert.ToInt32(this.config["max_total_frames"]))
        {
            Console.WriteLine("Reached designated end of run, sending shutdown message");
            this.isFinished = true;
        }

        return GetData();
    }

    public void DoneBatch()
    {
        this.totalBatches += 1;
        SaveCoreStats();
    }

    public void SaveCoreStats()
    {
        var statDict = new Dictionary<string, int>
        {
            { "steps", this.totalFrames },
            { "games", this.totalGames },
            { "batches", this.totalBatches }
        };
        File.WriteAllText(Path.Combine(this.logDirectory, "data.yaml"), new Serializer().Serialize(statDict));
    }

    public bool IsFinished()
    {
        return this.isFinished;
    }

    public List<Dictionary<string, object>> GetScores()
    {
        return this.gameStats;
    }

    public int GetTotalGames()
    {
        return this.totalGames;
    }

    public double GetElapsedTime()
    {
        return DateTime.Now.TimeOfDay.TotalSeconds - this.sessionStartTime;
    }
}
