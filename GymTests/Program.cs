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
    .WithNamingConvention(UnderscoredNamingConvention.Instance)  
    .Build();

Config config = deserializer.Deserialize<Config>(yamContent);
var env = new CarRacing(deviceType: DeviceType.CUDA, config.max_frames, config.image);
var effZero = new EfficientZero.Main();
effZero.Run(config, env, DeviceType.CUDA);

Console.WriteLine("Done");