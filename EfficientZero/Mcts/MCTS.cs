using DeepSharp.RL.Environs;
using EfficientZero.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace EfficientZero;

public class MCTS
{
    //torch.Device Device = new torch.Device(DeviceType.CUDA);

    public float K { get; set; }
    public Environ<Space, Space> Env { get; }
    private IEfficientZeroNet _net;
    private int _batchSize;
    private float _valWeight;
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
        _valWeight = config.val_weight;
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

    public TreeNode Search(int numberOfSimulations, Observation currentObs)
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
        LoadModel();

        _net.Eval();

        using (torch.no_grad())
        {
            int nodeCounter = 0;

            // Do the initial inference
            var (initialLatent, initialValue, initialPolicy) = _net.InitialInference(currentObs.Value.unsqueeze(0));
            float[] initPolicyProbs = torch.softmax(initialPolicy, 1).data<float>().ToArray();

            initPolicyProbs = Utils.AddDirichlet(initPolicyProbs, _alpha, K);

            // initialize the search tree with a root node
            var rootNode = new TreeNode(
                nodeCounter,
                latent: initialLatent,
                actionSize: _actionSize,
                valPrefix: 0,
                value: initialValue,
                polPred: initPolicyProbs,
                discount: _discount,
                minmax: _minMax,
                isRoot: true
            );

            // Update the count and value for the root node
            rootNode.UpdateVal(initialValue);
            rootNode.NumVisits++;
            rootNode.EstValueList[nodeCounter] = initialValue;
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

                    int action = currentNode.PickAction(currentNode.IsRoot); 

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
                        valuePrefixScalar = ScalarConverter.InverseScalarTransform(valuePrefix);
                        valueScalar = ScalarConverter.InverseScalarTransform(value);

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
        from the initialImage, and the actions.

        This unrolled training is how the dynamics function
        is trained - is it akin to training through a recurrent neural network with the prediction function
        as a head
        */

        _net.Train();

        Tensor totalLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalPolicyLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalValuePrefixLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        Tensor totalValueLoss = torch.tensor(0, ScalarType.Float32, device: _device);
        float totalConsistencyLoss = 0;

        for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++)
        {
            var batch = buffer.GetBatch(batchSize: _batchSize);

            Tensor batchPolicyLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            Tensor batchValuePrefixLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            Tensor batchValueLoss = torch.tensor(0, ScalarType.Float32, device: _device);
            float batchConsistencyLoss = 0;
            float batchWeight = 0;
            if (_config.priority_replay)
            {
                batchWeight = 0;
            }

            foreach (var batchData in batch)
            {
                if (_config.priority_replay)
                {
                    batchWeight += batchData.Weight;
                }

                int rolloutDepth = batchData.Actions.Count;

                //Tensor latent = _net.Represent(batchData.PreObs[0].Value.unsqueeze(0))[0];
                var (latent, initValueScalar, initPolicy) = _net.InitialInference(batchData.PreObs[0].Value.unsqueeze(0));
                float[] initPolicyProbs = torch.softmax(initPolicy, 1).data<float>().ToArray();
                var initValue = ScalarConverter.ScalarTransform(initValueScalar, halfWidth: _config.support_width).cuda();

                Tensor initTargetValue = torch.tensor(batchData.TargetValues[0], ScalarType.Float32, device: _device);
                Tensor initTargetPolicy = torch.tensor(batchData.TargetPolicies[0], ScalarType.Float32, device: _device);

                var initPredTargetValues = ScalarConverter.ScalarTransform(initTargetValue, halfWidth: _config.support_width).cuda();
                Tensor initPolicyLoss = _net.PolicyLoss.forward(initPolicy.squeeze(0), initTargetPolicy);
                Tensor initValueLoss = _net.ValueLoss.forward(initValue, initPredTargetValues);

                batchPolicyLoss += initPolicyLoss * batchData.Weight;
                batchValueLoss += initValueLoss * batchData.Weight;

                float grad = 1f;
                var rewardHidden = (torch.zeros(1, latent.unsqueeze(0).size(0), _lstmHiddenSize).cuda(),
                                    torch.zeros(1, latent.unsqueeze(0).size(0), _lstmHiddenSize).cuda());

                //then do all the rollouts
                for (int i = 1; i < rolloutDepth; i++)
                {
                    // We must do this sequentially, as the input to the dynamics function requires the output
                    // from the previous dynamics function
                    Tensor targetValue = torch.tensor(batchData.TargetValues[i], ScalarType.Float32, device: _device);
                    Tensor targetReward = torch.tensor(batchData.TargetRewards[i], ScalarType.Float32, device: _device);
                    Tensor targetPolicy = torch.tensor(batchData.TargetPolicies[i], ScalarType.Float32, device: _device);

                    var predTargetValues = ScalarConverter.ScalarTransform(targetValue, halfWidth: _config.support_width);
                    var predTargetValuePrefix = ScalarConverter.ScalarTransform(targetReward, halfWidth: _config.support_width);
                    var targetRew = targetReward.ToSingle();


                    var oneHotAction = nn.functional.one_hot(torch.from_array(new long[] { batchData.Actions[i].Value.ToInt64() }, device: _device), num_classes: _actionSize);

                    var (nextLatentState, valuePrefix, value, policy, reward) = _net.RecurrentInference(latent, oneHotAction, rewardHidden);
                    rewardHidden.Item1 = reward.Item1.unsqueeze(0);
                    rewardHidden.Item2 = reward.Item2.unsqueeze(0);

                    var policyProbabilities = torch.softmax(policy, 1).data<float>().ToArray();

                    var scalarValPrefix = ScalarConverter.InverseScalarTransform(valuePrefix);
                    var scalarValValue = ScalarConverter.InverseScalarTransform(value);
                    var targetVal = targetValue.ToSingle();

                    // We scale down the gradient, I believe so that the gradient at the base of the unrolled
                    // network converges to a maximum rather than increasing linearly with depth
                    nextLatentState.retain_grad();
                    grad *= .5f;
                    nextLatentState.grad = grad;

                    // The muzero paper calculates the loss as the squared difference between scalars
                    // but CrossEntropyLoss is used here for a more stable value loss when large values are encountered
                    Tensor policyLoss = _net.PolicyLoss.forward(policy.squeeze(0), targetPolicy);
                    Tensor valueLoss = _net.ValueLoss.forward(value.squeeze(0), predTargetValues);

                    //use k1_loss?
                    Tensor valuePrefixLoss = _net.RewardLoss.forward(valuePrefix.squeeze(0), predTargetValuePrefix);
                    Tensor? targetLatent = null;

                    if (_config.consistency_loss)
                    {
                        // This is the next latent state (i.e s+1)
                        float consistencyLoss = 0f;
                        targetLatent = _net.Represent(batchData.PreObs[i].Value.unsqueeze(0))[0].detach();
                        if (_config.image)
                        {
                            var dynamic_states_proj = _net.Projection(latent, true);
                            var gt_states_proj = _net.Projection(targetLatent, false);
                            var latentLoss = _net.ConsistencyLoss(dynamic_states_proj.squeeze(0), gt_states_proj.squeeze(0));
                            consistencyLoss = latentLoss.ToSingle();
                        }
                        else
                            consistencyLoss = _config.consistency_loss ? _net.ConsistencyLoss(nextLatentState.squeeze(0), targetLatent!).ToSingle() : 0f;

                        batchConsistencyLoss += consistencyLoss * batchData.Weight;
                    }

                    float pLoss = policyLoss.ToSingle();
                    float vLoss = valueLoss.ToSingle();
                    float rLoss = valuePrefixLoss.ToSingle();

                    //Console.WriteLine(string.Format("{0} {1} {2} {3}", pLoss, vLoss, rLoss, consistencyLoss));

                    batchPolicyLoss += policyLoss * batchData.Weight;
                    batchValueLoss += valueLoss * batchData.Weight;
                    batchValuePrefixLoss += valuePrefixLoss * batchData.Weight;


                    latent = nextLatentState;
                }
            }

            Tensor loss = (batchConsistencyLoss * _consistencyWeight + batchValueLoss * _valWeight + batchPolicyLoss + batchValuePrefixLoss);
            float lossVal = loss.ToSingle();
            if (_config.priority_replay)
            {
                float averageWeight = batchWeight / batch.Count;
                loss /= averageWeight;
            }

            _net.Optimizer.zero_grad();
            loss.backward();
            if (_config.grad_clip > 0)
            {
                torch.nn.utils.clip_grad_norm_(_net.parameters(), _config.grad_clip); // Assuming ClipGradNorm method exists
            }
            _net.Optimizer.step();

            totalLoss += loss;
            totalValueLoss += batchValueLoss;
            totalPolicyLoss += batchPolicyLoss;
            totalValuePrefixLoss += batchValuePrefixLoss;
            totalConsistencyLoss += batchConsistencyLoss;
        }

        totalLoss /= numberOfBatches;
        totalValueLoss /= numberOfBatches;
        totalPolicyLoss /= numberOfBatches;
        totalValuePrefixLoss /= numberOfBatches;
        totalConsistencyLoss /= numberOfBatches;

        var metricsDict = new Dictionary<string, float>
        {
            { "Loss/total", totalLoss.ToSingle() },
            { "Loss/policy", totalPolicyLoss.ToSingle() },
            { "Loss/valuePrefix", totalValuePrefixLoss.ToSingle() },
            { "Loss/value", totalValueLoss.ToSingle() * _valWeight },
            { "Loss/consistency", totalConsistencyLoss * _consistencyWeight }
        };

        this.SaveModel();

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
