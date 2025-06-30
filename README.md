# RLZero.Sharp


This is a C# implementation of EfficientZero using TorchSharp.<br>


##### Project
This is a personal project to recreate Muzero/EfficientZero in C#. It is solely for personal learning. 

## Environments
RLZero has been created to interact with multple environments. At first, it was created to complete LunarLander in Gym.Net. I believe there might be a few issues with the LunarLander environement in Gym.Net, specifically the ability to successfully land the lander. Although, this was in the early days of creating RLZero, so it could have been my code too. 

From there I moved onto cartpole. This environment was solved easily.

My ultimate goal was to make RLZero work with images instead of state vectors. So I decided to try with Gym's Car Racing environment. I recreated this game in MonoGame (TopDownCarPhysics). You can test the game by debugging the TopDownCarPhysics project specifically. The only part of the environment that is not implemented is the slip in the grass background. Everything else should be identical to Gym's Car Racing v3 environment.

Asteroids is another environment I created in MonoGame. The game code itself was created by someone else, then I added the environment later.

## Other projects
DeepSharp.RL is a nice project that hosts some basic RL learning models such as DQN. The only reason it is used in this project is for the Observation, Act, Reward, and Step classes. It also sets the base of the environment class. Otherwise, the agents, trainers, and replays are not used. DeepSharp.RL is reliant on DeepSharp.Utility (which is why it is also included).

Gym.Net is included for standarization when passing the Observation from MonoGame to RLZero.

## Getting Started
There are several projects in the solution. GymTest is the main project for testing RLZero. Program.cs holds the code for changing the environments. Right now it is set to CarRacing. The yaml files hold the configuration options for the environment/models.

If you want to see the observation image into the model, you can change ```config.render = True``` which will save a bitmap called car_race.bmp to your desktop. I would not recommend leaving this to true for the entire training or you will run into GDI problems. But it will work for a few episodes.

## Caveats
I was not able to translate the nonlinear gradient when training for the ```nextLatentState```. In MuZero/EfficientZero, they scale this tensor non-linearly. TorchSharp does not have a method for registering a hook, so for the time being, this is scaled linearly in the loss function.

Overall, I am unable to solve the Car Racing environment. The max score I can achieve is around 35 (the typical solveable score should be around 600). Thus, something is not working correctly and I still have problems with the code. Assistance with what I am doing wrong would be most appreciated. :-)

Right now I am focused on making the Car Racing environment work. I have not gone back to test cartpole/lunarlander/asteroids after updating much of the code. They will need some tweaking to work correctly, including referencing Environments, Rendering, and Winforms in Gym.Net's repository: https://github.com/SciSharp/Gym.NET/tree/master/src.

