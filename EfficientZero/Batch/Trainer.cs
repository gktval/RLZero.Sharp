using TorchSharp.Modules;
using TorchSharp;
using DeepSharp.RL.Environs;
using MathNet.Numerics;
using EfficientZero.Models;

namespace EfficientZero.Batch;

public class Trainer
{
    private DateTime lastTime;
    private dynamic config;
    private SummaryWriter writer;

    public Trainer()
    {
        lastTime = DateTime.Now;
    }

    public async Task<Dictionary<string, float>> Train(
        CartEfficientNet_Batch muNet,
        Memory memory,
        dynamic buffer,
        Dictionary<string, object> config,
        string logDir,
        torch.Device device,
        dynamic writer = null)
    {
        ///
        /// The train function simultaneously trains the prediction, dynamics and representation functions
        /// each batch has a series of values, rewards and policies, that must be predicted only
        /// from the initial_image, and the actions.

        /// This unrolled training is how the dynamics function
        /// is trained - is it akin to training through a recurrent neural network with the prediction function
        /// as a head
        ///

        this.config = config;
        torch.autograd.set_detect_anomaly(true);
        this.writer = new SummaryWriter(logDir);
        dynamic nextBatch = null;
        int totalBatches = await memory.GetDataAsync().Result["batches"];
        if (File.Exists(Path.Combine(logDir, "latest_model_dict.pt")))
        {
            muNet = await memory.LoadModelAsync(logDir, muNet);
        }
        muNet.To(device);

        while (await buffer.GetBufferLenAsync() == 0)
        {
            Thread.Sleep(1000);
        }
        double ms = DateTime.Now.TimeOfDay.TotalMilliseconds;
        var metricsDict = new Dictionary<string, float>();

        while (!memory.IsFinished())
        {
            PrintTiming("start");
            double st = DateTime.Now.TimeOfDay.TotalMilliseconds;

            float totalLoss = 0, totalPolicyLoss = 0, totalRewardLoss = 0, totalValueLoss = 0, totalConsistencyLoss = 0;
            if (nextBatch == null)
            {
                nextBatch = await buffer.GetBatchAsync(config["batch_size"], device);
            }
            PrintTiming("next batch command");
            float valDiff = 0;
            muNet.train();
            PrintTiming("to train");
            float batchPolicyLoss = 0, batchRewardLoss = 0, batchValueLoss = 0, batchConsistencyLoss = 0;
            PrintTiming("init");
            var batchData = await nextBatch;
            nextBatch = await buffer.GetBatchAsync(config["batch_size"]);
            PrintTiming("get batch");

            var images = batchData.images.To(device);
            var actions = batchData.actions.To(device);
            var targetValues = batchData.target_values.To(device);
            var targetRewards = batchData.target_rewards.To(device);
            var targetPolicies = batchData.target_policies.To(device);
            var weights = batchData.weights.To(device);
            PrintTiming("changing to device");

            System.Diagnostics.Debug.Assert(actions.le)

            var initImages = images[.., 0];
            PrintTiming("images0");
            var latents = muNet.Represent(initImages);
            PrintTiming("represent");
            dynamic outputHiddens = null;

            for (int i = 0; i < (int)config["rollout_depth"]; i++)
            {
                PrintTiming("rollout start");
                var screenT = torch.Tensor(depths) > i;
                if (torch.sum(screenT) < 1) continue;
                PrintTiming("for init");

                var targetValueStepI = targetValues[.., i];
                var targetRewardStepI = targetRewards[.., i];
                var targetPolicyStepI = targetPolicies[.., i];
                PrintTiming("make target");

                dynamic targetLatents = null;
                if (config.ContainsKey("consistency_loss"))
                {
                    targetLatents = muNet.Represent(images[:, i]).Detach();
                }
                PrintTiming("repreSENT");
                var oneHotActions = torch.nn.functional.one_hot(actions[.., i], numClasses: muNet.ActionSize).To(device);

            var predPolicyLogits, predValueLogits = muNet.Predict(latents);
            dynamic newLatents, predRewardLogits;

            if (config["value_prefix"])
            {
                (newLatents, predRewardLogits, outputHiddens) = muNet.Dynamics(latents, oneHotActions, outputHiddens);
            }
            else
            {
                (newLatents, predRewardLogits) = muNet.Dynamics(latents, oneHotActions);
            }
            PrintTiming("forward pass");
            newLatents.RegisterHook(grad => grad * 0.5);

            var predValues = SupportToScalar(Torch.Softmax(predValueLogits[screenT], dim: 1));
            var predRewards = SupportToScalar(Torch.Softmax(predRewardLogits[screenT], dim: 1));
            PrintTiming("support to scalar");
            var vvar = Torch.Var(predRewards);

            valDiff += Torch.Sum(targetValueStepI[screenT] - SupportToScalar(Torch.Softmax(predValueLogits[screenT], dim: 1))).Item<float>();
            var valLoss = new Torch.nn.MSELoss();
            var rewardLoss = new Torch.nn.MSELoss();
            var valueLoss = valLoss(predValues, targetValueStepI[screenT]);

            rewardLoss = rewardLoss(predRewards, targetRewardStepI[screenT]);
            var policyLoss = muNet.PolicyLoss(predPolicyLogits[screenT], targetPolicyStepI[screenT]);
            var consistencyLoss = config["consistency_loss"] ? muNet.ConsistencyLoss(latents[screenT], targetLatents[screenT]) : 0;

            batchPolicyLoss += (policyLoss * weights[screenT]).Mean();
            batchValueLoss += (valueLoss * weights[screenT]).Mean();
            batchRewardLoss += (rewardLoss * weights[screenT]).Mean();
            batchConsistencyLoss += (consistencyLoss * weights[screenT]).Mean();
            latents = newLatents;

            PrintTiming("done losses");
        }

        var batchLoss = batchPolicyLoss + batchRewardLoss + batchValueLoss * config["val_weight"] + batchConsistencyLoss * config["consistency_weight"];
        batchLoss = batchLoss.Mean();
        PrintTiming("batch loss");

        if (config["debug"])
        {
            Console.WriteLine($"v {batchValueLoss}, r {batchRewardLoss}, p {batchPolicyLoss}, c {batchConsistencyLoss}");
        }

        muNet.Optimizer.ZeroGrad();
        batchLoss.Backward();
        if (config["grad_clip"] != 0)
        {
            Torch.nn.Utils.ClipGradNorm(muNet.Parameters(), config["grad_clip"]);
        }
        muNet.Optimizer.Step();
        PrintTiming("optimizer");

        totalLoss += batchLoss.Item<float>();
        totalValueLoss += batchValueLoss;
        totalPolicyLoss += batchPolicyLoss;
        totalRewardLoss += batchRewardLoss;
        totalConsistencyLoss += batchConsistencyLoss;
        PrintTiming("loss");

        metricsDict = new Dictionary<string, float>();

        var frames = await memory.GetDataAsync().Result["frames"];
        if (totalBatches % 50 == 0)
        {
            await memory.SaveModelAsync(muNet.To(Torch.Device.CPU), logDir);
            muNet.To(device);
        }
        totalBatches++;

        if (writer != null)
        {
            foreach (var kvp in metricsDict)
            {
                writer.AddScalar(kvp.Key, kvp.Value, frames);
            }
        }

        if (totalBatches % 100 == 0)
        {
            Console.WriteLine($"Completed {totalBatches} total batches of size {config["batch_size"]}, last took {DateTime.Now.TimeOfDay.TotalMilliseconds - st:5.3}");
        }
        if (this.config["train_speed_profiling"])
        {
            Console.WriteLine($"WHOLE BATCH: {DateTime.Now.TimeOfDay.TotalMilliseconds - st}");
        }
        PrintTiming("saving/end");
    }

        return metricsDict;
    }

    private void PrintTiming(string stage)
    {
        // Placeholder for timing functionality
    }

    private dynamic SupportToScalar(dynamic tensor)
    {
        // Placeholder for support to scalar conversion
        return tensor;
    }
}
