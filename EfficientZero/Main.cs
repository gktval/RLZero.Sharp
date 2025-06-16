using DeepSharp.RL.Environs;
using TorchSharp;
using EfficientZero.ImageModels;
using EfficientZero.StateModels;

namespace EfficientZero;

public class Main
{
    public List<float> Run(Config config, Environ<Space, Space> env, DeviceType deviceType = DeviceType.CUDA)
    {
        // Create the directory to save models
        if (!Directory.Exists(config.log_dir))
            Directory.CreateDirectory(config.log_dir);

        if (config.log_name == "None")
            config.log_name = DateTime.Now.ToString("yyyy-MM-dd_HH.mm.ss");

        string logDirectory = Path.Combine(config.log_dir, config.log_name);

        int actionSize = env.ActionSpace.N;
        int observationSize = env.ObservationSpace.N;

        IEfficientZeroNet zeroNet = null;
        if (config.image)
            zeroNet = new EfficientZeroNet_Image(actionSize, config, deviceType);
        else
            zeroNet = new EfficientZeroNet_State(actionSize, observationSize, 51, config, deviceType);

        float learningRate = config.learning_rate;
        zeroNet.InitOptimizer(learningRate);

        var mcts = new MCTS(env, zeroNet, config);

        var memory = new ReplayBuffer(config);

        int totalGames = 0;
        var scores = new List<float>();

        while (totalGames < config.max_games)
        {
            int frames = 0;
            bool isGameOver = false;
            Observation frame = env.Reset();
            mcts.ResetMinMax();

            var gameRecord = new GameRecord(config, actionSize, frame, config.discount);

            float temperature = (config.max_games / 20f) / (totalGames + (config.max_games / 20f));
            //temperature = 0;
            float score = 0;

            if (totalGames % 10 == 0 && totalGames > 0)
            {
                learningRate *= config.learning_rate_decay;
                zeroNet.InitOptimizer(learningRate);
            }

            while (!isGameOver && frames < config.max_frames)
            {

                var node = mcts.Search(config.n_simulations, frame);

                Act action = node.PickBestAction(temperature);
                //int actKey = GetHumanAction();
                //Act action = new Act(torch.tensor(actKey));

                env.Render = config.render;

                Step step = env.Step(action, frames);
                frame = step.PostState;
                isGameOver = step.IsComplete;

                gameRecord.AddStep(step, node);

                frames++;
                score += step.Reward.Value;

                if (score < -100)
                    isGameOver = true;

            }

            gameRecord.AddPriorities();
            if (config.reanalyze)
            {
                memory.Reanalyse(mcts);
            }
            memory.SaveGame(gameRecord);

            var metricsDictionary = mcts.Train(memory, (config.n_batches));

            //TODO: Log the score

            scores.Add(score);
            float last100 = scores.Skip(Math.Max(scores.Count - 100, 0)).Take(100).Average();
            Console.WriteLine($"Completed game {totalGames + 1} with score {Math.Round(score, 2)}. Loss was {Math.Round(metricsDictionary["Loss/total"], 2)}. Rolling average is {Math.Round(last100, 1)}");
            totalGames++;
        }

        env.Close();
        return scores;
    }

    private int GetHumanAction()
    {
        var key = Console.ReadKey().Key;

        if (key == ConsoleKey.LeftArrow)
            return 1;
        else if (key == ConsoleKey.RightArrow)
            return 2;
        else if (key == ConsoleKey.UpArrow)
            return 3;
        else if (key == ConsoleKey.DownArrow)
            return 4;

        return 0;
    }
}
