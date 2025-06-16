// See https://aka.ms/new-console-template for more information
using TorchSharp;
using LunarLander;
using System;
using YamlDotNet.Serialization.NamingConventions;
using YamlDotNet.Serialization;
using EfficientZero;
using static TorchSharp.torch;

Console.WriteLine("Initializing Car Racing");

// Set the device to the first GPU
torch.InitializeDevice(new torch.Device(DeviceType.CUDA));

//string yamlFile = "config-lunarlander.yaml";
string yamlFile = "config-carracing.yaml";
string yamContent = File.ReadAllText(yamlFile);

var deserializer = new DeserializerBuilder()
    .WithNamingConvention(UnderscoredNamingConvention.Instance)  // see height_in_inches in sample yml 
    .Build();

Config config = deserializer.Deserialize<Config>(yamContent);
var env = new CarRacing(deviceType: DeviceType.CUDA, config.max_frames, config.image);
var effZero = new EfficientZero.Main();
effZero.Run(config, env, DeviceType.CUDA);


//float k = (float)Math.Pow(3, .5);
//var model = new DPW(0.25f, 0.2f, env.Observation, env, k);
//bool isDone = false;
//while (!isDone)
//{
//    model.Learn(500);
//    var action = model.BestActionReward();
//    var step = env.Step(action, env.Epoch);
//    isDone = step.IsComplete;
//    model.Forward(action, step.PostState);

//    Console.WriteLine(string.Format("{0}", step.Reward));
//}

//isDone = false;
//var actionList = env.GetActions();
//env.Reset();
//env.Render = true;
//int epoch = 0;
//float reward = 0;
//while(!isDone)
//{
//    var step = env.Step(actionList[epoch], epoch);
//    reward = step.Reward.Value;

//    await Task.Delay(20);
//    isDone = env.IsDone;
//    epoch++;
//}

//if (reward > 0)
//    Console.WriteLine("You Win!");

Console.WriteLine("Done");