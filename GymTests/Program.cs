using TorchSharp;
using YamlDotNet.Serialization.NamingConventions;
using YamlDotNet.Serialization;
using EfficientZero;
using GymTest;

//Console.WriteLine("Initializing Asteroids");
Console.WriteLine("Initializing Car Racing");

// Set the device to the first GPU
torch.InitializeDevice(new torch.Device(DeviceType.CUDA));

//string yamlFile = "config-lunarlander.yaml";
string yamlFile = "config-carracing.yaml";
//string yamlFile = "config-asteroids.yaml";
string yamContent = File.ReadAllText(yamlFile);

var deserializer = new DeserializerBuilder()
    .WithNamingConvention(UnderscoredNamingConvention.Instance)  
    .Build();

Config config = deserializer.Deserialize<Config>(yamContent);
var env = new CarRacing(deviceType: DeviceType.CUDA, config.max_frames, config.image, config.frame_skip);
var effZero = new EfficientZero.Main();
effZero.Run(config, env, DeviceType.CUDA);

Console.WriteLine("Done");