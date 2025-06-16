using ACadSharp.Entities;
using EfficientZero.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace EfficientZero;

public class Reanalyzer
{
    private Device _device;
    private Dictionary<string, object> _config;
    private string _logDirectory;

    public Reanalyser(Dictionary<string, object> config, string logDirectory, Device device)
    {
        _device = device;
        _config = config;
        _logDirectory = logDirectory;
    }

    public void Run(CartEfficientNet_Batch muNet, Memory memory, Buffer buffer)
    {
        while (!memory.IsFinished())
        {
            if (Directory.Exists(_logDirectory) && Directory.GetFiles(_logDirectory).Contains("latest_model_dict.pt"))
            {
                muNet = memory.LoadModel(_logDirectory, muNet);
                muNet.to(_device);
            }

            // No point reanalysing until there are multiple games in the history
            while (true)
            {
                int bufferLength = buffer.GetBufferLength();
                var trainStats = memory.GetData();
                int currentGame = trainStats["games"];
                if (bufferLength >= 1 && currentGame >= 2)
                {
                    break;
                }

                Thread.Sleep(1000);
            }

            muNet.train();
            muNet = muNet.To(_device);

            float[] probabilities = buffer.GetReanalyseProbabilities();

            if (probabilities.Length > 0)
            {
                int[] indices = buffer.GetBufferIndices();
                int selectedIndex = -1;

                try
                {
                    selectedIndex =Utils.GetRandomFromProb(indices, probabilities);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(string.Join(", ", probabilities) + " " + string.Join(", ", indices));
                }

                var gameRecord = buffer.GetBufferIndex(selectedIndex);
                var minmax = memory.GetMinMax();

                var values = gameRecord.Values;

                for (int i = 0; i < gameRecord.Observations.Length - 1; i++)
                {
                    float observation;
                    if (_config["obs_type"].ToString() == "image")
                    {
                        observation = gameRecord.GetLastN(pos: i);
                    }
                    else
                    {
                        observation = (gameRecord.Observations[i] - 128) / 64;
                        //observation = ConvertFromInt(gameRecord.Observations[i], config["obs_type"].ToString());
                    }

                    var newRoot = Search(_config, muNet, observation, minmax, _logDirectory, this._device);
                    values[i] = newRoot.AverageValue;
                }

                buffer.UpdateValues(selectedIndex, values);
                buffer.AddPriorities(selectedIndex, reanalysing: true);
                Console.WriteLine($"Reanalysed game {selectedIndex}");
            }
            else
            {
                Thread.Sleep(5000);
            }
        }
    }
}
