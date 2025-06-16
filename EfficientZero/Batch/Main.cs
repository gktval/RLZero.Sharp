using TorchSharp.Modules;
using TorchSharp;
using YamlDotNet.Serialization;
using DeepSharp.RL.Environs;
using EfficientZero.Models;

namespace EfficientZero.Batch;

public class Main
{
    public static List<double> Run(Dictionary<string, object> config, Environ<Space, Space> env, bool trainOnly = false)
    {
        int actionSize = env.ActionSpace.N;
        config["action_size"] = actionSize;
        Console.WriteLine($"Action size: {actionSize}");

        int obsSize = env.ObservationSpace.N;
        Console.WriteLine($"Observation size: {obsSize}");

        var muZeroNetwork = new CartEfficientNet_Batch(actionSize, obsSize, config);
        muZeroNetwork.InitOptimizer((double)config["learning_rate"]);

        if ((string)config["log_name"] == "last")
        {
            var runs = Directory.GetFiles((string)config["log_dir"], $"{config["env_name"]}*").ToList();
            if (runs.Any())
            {
                config["log_name"] = runs.OrderBy(x => x).Last();
            }
            else
            {
                config["log_name"] = "None";
            }
        }
        if ((string)config["log_name"] == "None")
        {
            config["log_name"] = $"{DateTime.Now:yyyy-MM-dd_HH.mm.ss}{config["env_name"]}";
        }

        Directory.CreateDirectory("buffers");

        Console.WriteLine($"Logging to '{config["log_name"]}'");

        string logDirectory = Path.Combine((string)config["log_dir"], (string)config["log_name"]);

        if (!Directory.Exists(logDirectory))
        {
            Directory.CreateDirectory(logDirectory);
        }
        if (!File.Exists(Path.Combine(logDirectory, "data.yaml")))
        {
            var initDict = new Dictionary<string, int> { { "games", 0 }, { "steps", 0 }, { "batches", 0 } };
            var serializer = new Serializer();
            File.WriteAllText(Path.Combine(logDirectory, "data.yaml"), serializer.Serialize(initDict));
        }

        var tbWriter = new SummaryWriter(logDirectory);
        var workers = new List<Task>();

        bool useCuda = (bool)config["try_cuda"] && torch.cuda.is_available();

        double bufferGpus = useCuda ? 0.1 : 0;
        var memory = new Memory(config, logDirectory);
        var buffer = new Buffer(config, memory);

        DateTime startTime = DateTime.Now;
        var scores = new List<double>();

        var device = new torch.Device(useCuda ? "cuda:0" : "cpu");
        Console.WriteLine($"Training on device: {device}");

        var player = Player.Options(numCpus: 0.3).Remote(logDirectory: logDirectory);

        double trainCpus = useCuda ? 0 : 0.1;
        double trainGpus = useCuda ? 0.9 : 0;
        var trainer = new Trainer();

        if (!trainOnly)
        {
            workers.Add(player.Play.Remote(config, muNet: muZeroNetwork, logDir: logDirectory, device: torch.Device("cpu"), memory: memory, buffer: buffer, env: environment));
        }

        workers.Add(trainer.Train(muZeroNetwork, memory, buffer, config, logDirectory, device));

        if ((bool)config["reanalyse"])
        {
            Console.WriteLine("Adding reanalyser");
            var analyser = Reanalyser.Options(numCpus: 0.1).Remote(config, logDir: logDirectory);
            workers.Add(analyser.Reanalyse.Remote(muNet: muZeroNetwork, memory: memory, buffer: buffer));
        }

        Task.WhenAll(workers).Wait();

        env.Close();
        scores = memory.GetScores.Remote().GetAwaiter().GetResult();
        return scores;
    }
}
