using DeepSharp.RL.Environs;
using EfficientZero.Models;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace EfficientZero;

public class MCTS
{
    //torch.Device Device = new torch.Device(DeviceType.CUDA);

    public float K { get; set; }
    public Environ<Space, Space> Env { get; }
    private IEfficientZeroNet _net;
    private int _batchSize;
    private float _valueWeight;
    private float _consistencyWeight;
    private float _discount;
    private float _alpha;
    private MinMax _minMax;
    private Config _config;
    private string _logDir;
    private int _actionSize;
    private Device _device;
    private int _lstmHiddenSize;

    /// <summary>
    /// Base class for MCTS based on Monte Carlo Tree Search for Continuous and Stochastic Sequential
    /// Decision Making Problems, Courtoux
    /// </summary>
    /// <param name="initial_obs">initial state of the tree.Returned by env.reset()</param>
    /// <param name="env">Environment</param>
    /// <param name="k">exporation parameter of UCB</param>
    public MCTS(Environ<Space, Space> env, IEfficientZeroNet model, Config config, DeviceType deviceType = DeviceType.CUDA)
    {
        Env = env;
        _device = new torch.Device(deviceType);
        _net = model;
        _config = config;

        _actionSize = env.ActionSpace.N;

        this.K = config.explore_frac;
        _batchSize = config.batch_size;
        _valueWeight = config.value_weight;
        _consistencyWeight = config.consistency_weight;
        _discount = config.discount;
        _alpha = config.root_dirichlet_alpha;
        _logDir = config.log_dir;
        _lstmHiddenSize = config.lstm_hidden_size;

        // keeps track of the highest and lowest values found
        _minMax = new MinMax();

    }

    public void ResetMinMax()
    {
        _minMax = new MinMax();
    }

    public TreeNode Search(int numberOfSimulations, Observation currentObs, bool addNoise)
    {
        /*
        This function takes a frame and creates a tree of possible actions that could
        be taken from the frame, assessing the expected value at each location
        and returning this tree which contains the data needed to choose an action

        The models expect inputs where the first dimension is a batch dimension,
        but the way in which we traverse the tree means we only pass a single
        input at a time. There is therefore the need to consistently squeeze and unsqueeze
        (ie add and remove the first dimension) so as not to confuse things by carrying around
        extraneous dimensions

        Note that mu_net returns logits for the policy, value and reward
        and that the value and reward are represented categorically rather than
        as a scalar
        */

        //LoadModel();

        _net.Eval();

        using (torch.no_grad())
        {
            int nodeCounter = 0;

            // Do the initial inference
            var (initialLatent, initialValue, initialPolicy) = _net.InitialInference(currentObs.Value.unsqueeze(0));
            float[] initPolicyProbs = torch.softmax(initialPolicy, 1).data<float>().ToArray();
            var scalarValue = ScalarConverter.InverseScalarTransform(initialValue).squeeze(0).ToSingle();

            if (addNoise)
                initPolicyProbs = Utils.AddDirichlet(initPolicyProbs, _alpha, K);

            // initialize the search tree with a root node
            var rootNode = new TreeNode(
                nodeCounter,
                latent: initialLatent,
                actionSize: _actionSize,
                valPrefix: 0,
                value: scalarValue,
                polPred: initPolicyProbs,
                discount: _discount,
                minmax: _minMax,
                isRoot: true
            );

            // Update the count and value for the root node
            rootNode.UpdateVal(rootNode.Value);
            rootNode.NumVisits++;
            rootNode.EstValueList[nodeCounter] = rootNode.Value;
            nodeCounter += 1;

            rootNode.HiddenReward = (torch.zeros(1, initialLatent.size(0), _lstmHiddenSize).cuda(),
                                torch.zeros(1, initialLatent.size(0), _lstmHiddenSize).cuda());

            for (int i = 0; i < numberOfSimulations; i++)
            {
                // vital to have to grad or the size of the computation graph quickly becomes gigantic
                var currentNode = rootNode;
                bool isNewNode = false;
                float valueScalar = 0f;
                float valuePrefixScalar = 0f;

                // search list tracks the route of the simulation through the tree
                var searchList = new List<TreeNode> { currentNode };
                while (!isNewNode)
                {
                    var policyPred = currentNode.PolPred;
                    var latent = currentNode.Latent;
                    var rewardHidden = currentNode.HiddenReward;

                    int action = currentNode.SelectChild(currentNode.IsRoot);

                    // if we pick an action that's been picked before we don't need to run the model to explore it
                    if (currentNode.Children[action] == null)
                    {
                        // Convert to a 2D tensor one-hot encoding the action
                        var oneHotAction = nn.functional.one_hot(torch.from_array(new long[] { action }, device: _device), num_classes: _actionSize);

                        // apply the dynamics function to get a representation of
                        // the state after the action, and the reward gained
                        // then estimate the policy and value at this new state
                        var (nextLatentState, valuePrefix, value, policy, reward) = _net.RecurrentInference(latent, oneHotAction, rewardHidden);
                        //update the hidden reward states
                        rewardHidden.Item1 = reward.Item1.unsqueeze(0);
                        rewardHidden.Item2 = reward.Item2.unsqueeze(0);

                        // convert logits to scalars and probability distributions
                        //var rewardScalarC = ScalarConverter.InverseScalarTransform(reward.Item1);
                        //var rewardScalarH = ScalarConverter.InverseScalarTransform(reward.Item2);
                        valuePrefixScalar = ScalarConverter.InverseScalarTransform(valuePrefix).squeeze(0).ToSingle();
                        valueScalar = ScalarConverter.InverseScalarTransform(value).squeeze(0).ToSingle();

                        var policyProbabilities = torch.softmax(policy, 1).data<float>().ToArray();

                        currentNode.Insert(
                            nodeCounter,
                            actionIndex: action,
                            latent: nextLatentState,
                            valPrefix: valuePrefixScalar,
                            value: valueScalar,
                            polPred: policyProbabilities,
                            reward: 0,
                            hiddenReward: rewardHidden,
                            minmax: _minMax,
                            targetPrefixValue: valueScalar
                        );
                        nodeCounter += 1;

                        // We have reached a new node and therefore this is the end of the simulation
                        isNewNode = true;
                    }

                    currentNode = currentNode.Children[action];

                    searchList.Add(currentNode);
                }

                // Updates the visit counts and average values of the nodes that have been traversed
                Backpropagate(searchList, valueScalar);
            }

            rootNode.Scores = rootNode.GetSearchPolicy();
            return rootNode;
        }
    }

    public void Backpropagate(List<TreeNode> searchList, float value)
    {
        int leafNodeId = searchList[searchList.Count - 1].Id;
        float bootstrapValue = value;
        // Going backward through the visited nodes, we increase the visit count of each by one
        // and set the value, discounting the value at the node ahead, but then adding the reward
        for (int i = searchList.Count - 1; i >= 0; i--)
        {
            TreeNode node = searchList[i];
            node.EstValueList[leafNodeId] = bootstrapValue;
            node.UpdateVal(bootstrapValue);
            node.NumVisits += 1;

            //get the new updated value for the parent node
            bootstrapValue = node.GetReward() + _discount * bootstrapValue;
            _minMax.Update(bootstrapValue);
        }
    }

    public Dictionary<string, float> Train(ReplayBuffer buffer, int numberOfBatches)
    {
        /*
        The train function simultaneously trains the prediction, dynamics and representation functions
        each batch has a series of values, rewards and policies, that must be predicted only
        from the initial Observation, and the actions.

        This unrolled training is how the dynamics function
        is trained - is it akin to training through a recurrent neural network with the prediction function
        as a head
        */

        _net.Train();

        Tensor totalLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalPolicyLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalValuePrefixLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalValueLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalConsistencyLoss = torch.tensor(0, ScalarType.Float32, device: _device);

        for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++)
        {
            List<BufferData> batch = buffer.GetBatch(batchSize: _batchSize);
            // batch everything up
            List<Tensor> listPreObs = new List<Tensor>();
            List<Tensor> listTargetPolicy = new List<Tensor>();
            List<Tensor> listTargetVals = new List<Tensor>();
            List<Tensor> listTargetPrefix = new List<Tensor>();
            List<Tensor> listActions = new List<Tensor>();

            for (int i = 0; i < 6; i++)
            {
                List<Tensor> bPreObs = new List<Tensor>();
                List<Tensor> bTargetPolicy = new List<Tensor>();
                List<Tensor> bTargetVals = new List<Tensor>();
                List<Tensor> bTargetPrefix = new List<Tensor>();
                List<Tensor> bActions = new List<Tensor>();

                foreach (var batchData in batch)
                {
                    if (batchData.PreObs.Count > 1 && batchData.PreObs.Count > i) //greater than one. Otherwise there are no rollouts
                    {
                        bPreObs.Add(batchData.PreObs[i].Value);

                        bTargetPolicy.Add(torch.tensor(batchData.TargetPolicies[i], ScalarType.Float32, device: _device));

                        Tensor iValue = torch.tensor(batchData.TargetValues[i], ScalarType.Float32, device: _device);
                        Tensor scalarTargetValues = ScalarConverter.ScalarTransform(iValue, halfWidth: _config.support_width).cuda();
                        bTargetVals.Add(scalarTargetValues);

                        Tensor iReward = torch.tensor(batchData.TargetRewards[i], ScalarType.Float32, device: _device);
                        Tensor scalarTargetRewards = ScalarConverter.ScalarTransform(iReward, halfWidth: _config.support_width).cuda();
                        bTargetPrefix.Add(scalarTargetRewards);

                        var oneHotAction = nn.functional.one_hot(torch.from_array(new long[] { batchData.Actions[i].Value.ToInt64() }, device: _device), num_classes: _actionSize);
                        bActions.Add(oneHotAction.squeeze(0));
                    }
                }
                listPreObs.Add(torch.stack(bPreObs));
                listTargetPolicy.Add(torch.stack(bTargetPolicy));
                listTargetVals.Add(torch.stack(bTargetVals));
                listTargetPrefix.Add(torch.stack(bTargetPrefix));
                listActions.Add(torch.stack(bActions));
            }

            List<float> bWeights = new List<float>();
            for (int i = 0; i < batch.Count; i++)
                if (batch[i].Actions.Count > 1)
                    bWeights.Add(batch[i].Weight);
            var weights = torch.from_array(bWeights.ToArray(), device: _device);

            Tensor batchPolicyLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            Tensor batchValuePrefixLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            Tensor batchValueLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            Tensor batchConsistencyLoss = torch.tensor(0, ScalarType.Float32, device: _device);

            var (latent, initValue, initPolicy) = _net.InitialInference(listPreObs[0]);


            // --- Batch ---
            //Tensor initPolicyLoss = -(F.log_softmax(initPolicy, 1) * listTargetPolicy[0]).sum(1);
            //Tensor initValueLoss = -(F.log_softmax(initValue, 1) * listTargetVals[0]).sum(1);
            //var sPolicyLoss = initPolicyLoss.data<float>();
            //var sValueLoss = initValueLoss.data<float>();

            // --- Test CrossEntropy ---
            Tensor initValueLoss = _net.ValueLoss.forward(initValue, listTargetVals[0]);
            Tensor initPolicyLoss = _net.PolicyLoss.forward(initPolicy.squeeze(0), listTargetPolicy[0]);
            //var sPolicyLoss = initPolicyLoss.data<float>();
            //var sValueLoss = initValueLoss.data<float>();

            batchPolicyLoss += initPolicyLoss;
            batchValueLoss += initValueLoss;

            float grad = 1f;
            var rewardHidden = (torch.zeros(1, latent.size(0), _lstmHiddenSize).cuda(),
                                torch.zeros(1, latent.size(0), _lstmHiddenSize).cuda());

            int rolloutDepth = 6;
            //then do all the rollouts
            for (int i = 1; i < rolloutDepth; i++)
            {
                if (latent.size(0) > listActions[i].size(0))
                {
                    int newSize = (int)listActions[i].size(0);
                    latent = latent[..newSize];
                    rewardHidden.Item1 = rewardHidden.Item1[.., ..newSize, ..];
                    rewardHidden.Item2 = rewardHidden.Item2[.., ..newSize, ..];
                }
                var (nextLatentState, valuePrefix, value, policy, reward) = _net.RecurrentInference(latent, listActions[i], rewardHidden);
                rewardHidden.Item1 = reward.Item1.unsqueeze(0);
                rewardHidden.Item2 = reward.Item2.unsqueeze(0);

                // We scale down the gradient, I believe so that the gradient at the base of the unrolled
                // network converges to a maximum rather than increasing linearly with depth

                // ------------How to do this in c#??? --------------------
                // nextLatentState.register_hook(f=> f.grad * 0.5f);
                // --------------------------------------------------------

                // The muzero paper calculates the loss as the squared difference between scalars
                // but CrossEntropyLoss is used here for a more stable value loss when large values are encountered

                //Tensor policyLoss = -(F.log_softmax(policy, 1) * listTargetPolicy[i]).sum(1);
                //Tensor valueLoss = -(F.log_softmax(value, 1) * listTargetVals[i]).sum(1);
                //Tensor valuePrefixLoss = -(F.log_softmax(valuePrefix, 1) * listTargetPrefix[i]).sum(1);
                Tensor policyLoss = _net.PolicyLoss.forward(policy.squeeze(0), listTargetPolicy[i]);
                Tensor valueLoss = _net.ValueLoss.forward(value.squeeze(0), listTargetVals[i]);
                Tensor valuePrefixLoss = _net.RewardLoss.forward(valuePrefix.squeeze(0), listTargetPrefix[i]);


                if (_config.consistency_loss)
                {
                    // This is the next latent state (i.e s+1)
                    Tensor consistencyLoss = torch.zeros(nextLatentState.size(0));
                    Tensor targetLatent = _net.Represent(listPreObs[i]);
                    if (_config.image)
                    {
                        var proj_state = nextLatentState.reshape(-1, 2304);
                        var target_proj_state = targetLatent.reshape(-1, 2304);
                        //var proj_state = _net.Projection(nextLatentState, true);
                        //var target_proj_state = _net.Projection(targetLatent, false);

                        var latentLoss = _net.ConsistencyLoss(proj_state, target_proj_state);

                        consistencyLoss = latentLoss;
                    }
                    else
                        consistencyLoss =  _net.ConsistencyLoss(nextLatentState.squeeze(0), targetLatent!).ToSingle();

                    if (batchConsistencyLoss.ndim > 0 && batchConsistencyLoss.size(0) != consistencyLoss.size(0))
                        consistencyLoss = nn.functional.pad(consistencyLoss, pad: [0, batchConsistencyLoss.size(0) - consistencyLoss.size(0)], mode: PaddingModes.Constant, value: 0);
                    batchConsistencyLoss += consistencyLoss;
                }

                if (batchPolicyLoss.ndim > 0 && batchPolicyLoss.size(0) != policyLoss.size(0))
                {
                    int pad = (int)(batchPolicyLoss.size(0) - policyLoss.size(0));
                    policyLoss = nn.functional.pad(policyLoss, pad: [0, pad], mode: PaddingModes.Constant, value: 0);
                    valueLoss = nn.functional.pad(valueLoss, pad: [0, pad], mode: PaddingModes.Constant, value: 0);
                    valuePrefixLoss = nn.functional.pad(valuePrefixLoss, pad: [0, pad], mode: PaddingModes.Constant, value: 0);
                }
                batchPolicyLoss += policyLoss;
                batchValueLoss += valueLoss;
                batchValuePrefixLoss += valuePrefixLoss;

                latent = nextLatentState;
            }

            float bpLoss = batchPolicyLoss.ToSingle();
            float bvLoss = batchValueLoss.ToSingle();
            float brLoss = batchValuePrefixLoss.ToSingle();

            Tensor loss = (batchConsistencyLoss * _consistencyWeight + batchValueLoss * _valueWeight + batchPolicyLoss + batchValuePrefixLoss);
            loss = (loss * weights).mean();
            float lossVal = loss.ToSingle();

            _net.Optimizer.zero_grad();
            List<Tensor> grads = [torch.tensor(1.0f / _config.rollout_depth)];

            loss.backward(grads);
            //loss.backward();

            if (_config.grad_clip > 0)
                torch.nn.utils.clip_grad_norm_(_net.parameters(), _config.grad_clip);
            _net.Optimizer.step();

            totalLoss += loss;
        }

        totalLoss /= numberOfBatches;

        var metricsDict = new Dictionary<string, float>
        {
            { "Loss/total", totalLoss.ToSingle() },
        };

        //this.SaveModel();

        return metricsDict;

    }


    public void SaveModel()
    {
        string location = Path.Combine(_logDir, "latest_model_dict.pt");
        _net.Save(location);
    }

    public void LoadModel()
    {
        string location = Path.Combine(_logDir, "latest_model_dict.pt");
        if (File.Exists(location))
        {
            _net.Load(location, strict: true);
        }
    }
}
