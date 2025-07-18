﻿using System.Threading.Tasks;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

// ReSharper disable once CheckNamespace
namespace Gym.Envs {
    public interface IEnv {
        Dict Metadata { get; set; }
        (float From, float To) RewardRange { get; set; }
        Space ActionSpace { get; set; }
        Space ObservationSpace { get; set; }
        NDArray Reset();
        Step Step(object action);
        Task<Step> StepAsync(object action);
        void CloseEnvironment();
        void Seed(int seed);
    }
}