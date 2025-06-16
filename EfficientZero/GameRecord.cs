
using DeepSharp.RL.Environs;

namespace EfficientZero;

public class GameRecord
{
    // This class stores the relevant history of a single game
    public GameRecord(Config config, int actionSize, Observation initObs, float discount = 0.8f)
    {
        Config = config;
        ActionSize = actionSize;  // Number of available actions
        Discount = discount;  // Discount rate to be applied to future rewards

        // List of states received from the game
        PreObs = new List<Observation>();
        // List of states received from the game
        // List of actions taken in the game
        Actions = new List<Act>();
        // List of estimated future rewards after a step
        Rewards = new List<float>();
        // List of the number of times each possible action was sampled at the root of the search tree
        SearchStats = new List<List<int>>();
        // Estimated sum of rewards up to a step from the search
        Values = new List<float>();
        // Values predicted by the initial observation
        PredValues = new List<float>();
        Priorities = new List<float>();
    }

    public Config Config { get; }
    public int ActionSize { get; }
    public float Discount { get; }
    public List<Observation> PreObs { get; }
    public List<Act> Actions { get; }
    public List<float> Rewards { get; }
    public List<List<int>> SearchStats { get; }
    /// <summary>
    /// Value predicted by the search
    /// </summary>
    public List<float> Values { get; }
    /// <summary>
    /// Value predicted by the initial observation
    /// </summary>
    public List<float> PredValues { get; }
    public List<float> Priorities { get; }

    public void AddStep(Step step, TreeNode root)
    {
        // Root is a TreeNode object at the root of the search tree for the given state

        // Note that when taking a step you get the action, reward and new observation
        // but for training purposes we want to connect the reward with the action and *old* observation.
        // We therefore add the first frame when we initialize the class, so connected frame-action-reward
        // tuples have the same index

        PreObs.Add(step.PreState);
        Actions.Add(step.Action);
        Rewards.Add(step.Reward.Value);
        if (root.Children != null)
        {
            List<int> childVisits = new List<int>();
            foreach (var child in root.Children)
            {
                if (child is not null)
                    childVisits.Add(child.NumVisits);
                else
                    childVisits.Add(0);
            }

            SearchStats.Add(childVisits);
        }

        Values.Add(root.AverageVal);
        PredValues.Add(root.Value);
    }

    public void AddPriorities()
    {
        //Priorities.Clear();
        if (Priorities.Count != 0)
            throw new System.Exception("Priorities should be empty initially.");

        List<float> realValue = new List<float>();
        int rewardDepth = 25;
        for (int index = 0; index < Values.Count; index++)
        {
            int bootstrapIndex = index + rewardDepth;
            int lastBootstrapIndex = Math.Min(bootstrapIndex, Rewards.Count - 1);

            float targetSum = Values[index] * (float)Math.Pow(Discount, lastBootstrapIndex);
            for (int j = 0; j < rewardDepth; j++)
                if (index + j < Rewards.Count - 1) // -1 because it is episodic
                    targetSum += Rewards[index + j] * (float)Math.Pow(Discount, j);

            //float priority = (Values[index] - targetSum) * (Values[index] - targetSum);
            float priority = (PredValues[index] - Values[index]) * (PredValues[index] - Values[index]);
            priority = (float)Math.Max(priority, 1);
            Priorities.Add(priority);
        }
    }

    public BufferData MakeTarget(int index, int rewardDepth = 5, int rolloutDepth = 3)
    {
        // index is where in the record of the game we start
        // rewardDepth is how far into the future we use the actual reward - beyond this we use predicted value

        // rolloutDepth is how many iterations of the dynamics function we take, from where we apply the represent function
        // it acts like an additional dimension of batching when we make the target but the crucial difference is that
        // when we train, we will use a hidden state by repeated application of the dynamics function from the initImage
        // rather than creating a new hidden state from the game obs at the time
        // this is necessary to train the dynamics function to give useful information for predicting the value

        List<float> targetValuePrefixes = new List<float>();
        List<float> targetValues = new List<float>();
        List<List<float>> targetPolicies = new List<List<float>>();

        int gameLength = SearchStats.Count;

        // Make sure we don't try to roll out beyond end of game
        rolloutDepth = System.Math.Min(rolloutDepth, gameLength - index);

        float valuePrefix = 0f;
        for (int i = 0; i < rolloutDepth; i++)
        {
            // If we have an estimated value at the current index + rewardDepth
            // then this is our base value (after discounting)
            // else we start at 0 
            int bootstrapIndex = index + rewardDepth + i;
            valuePrefix += Rewards[index + i];
            int lastBootstrapIndex = Math.Min(bootstrapIndex, Rewards.Count - 1);

            float targetSum = Values[index + i] * (float)Math.Pow(Discount, lastBootstrapIndex);
            for (int j = 1; j < rewardDepth; j++) //+1 because we want future rewards
                if (index + i + j < Rewards.Count - 1) // -1 because it is episodic
                    targetSum += Rewards[index + i + j] * (float)Math.Pow(Discount, j);

            targetValues.Add(targetSum);
            targetValuePrefixes.Add(valuePrefix);

            int totalSearches = 0;
            foreach (var searches in SearchStats[index + i])
            {
                totalSearches += searches;
            }

            // The target policy is the fraction of searches which went down each action at the root of the tree
            List<float> policy = new List<float>();
            foreach (var search in SearchStats[index + i])
            {
                policy.Add(search / (float)totalSearches);
            }
            targetPolicies.Add(policy);
        }

        // include all observations for consistency loss
        List<Observation> preObs = PreObs.GetRange(index, rolloutDepth);
        List<Act> actions = Actions.GetRange(index, rolloutDepth);
        //List<Act> hiddenRewards = .GetRange(index, rolloutDepth);

        return new BufferData(preObs, actions, targetValues, targetValuePrefixes, targetPolicies);
    }

    public GameRecord Reanalyse(MCTS mcts)
    {
        int numSim = Config.n_simulations;
        for (int i = 0; i < PreObs.Count - 1; i++)
        {
            TreeNode node = mcts.Search(numSim, PreObs[i]);

            // In the paper, it says they recompute trajectories using the "mean" value at the root node.
            Values[i] = node.AverageVal;
            PredValues[i] = node.Value;

            if (node.Children != null)
            {
                List<int> childVisits = new List<int>();
                foreach (var child in node.Children)
                {
                    if (child is not null)
                        childVisits.Add(child.NumVisits);
                    else
                        childVisits.Add(0);
                }

                SearchStats[i] = childVisits;
            }
        }

        //Refresh the priorities
        Priorities.Clear();
        AddPriorities();

        return this;
    }
}
