using DeepSharp.RL.Environs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace EfficientZero;

/// <summary>
/// TreeNode is an individual node of a search tree.
/// It has one potential child for each potential action which, if it exists, is another TreeNode
/// Its function is to hold the relevant statistics for deciding which action to take.
/// </summary>
public class TreeNode
{
    /// <summary>
    /// An arbitrary id of the node. Not really needed...
    /// </summary>
    public int Id { get; private set; }
    /// <summary>
    /// Keeps track of the nodes subsequent children ValuePrefix
    /// This property is not used and is only for debugging purposes
    /// </summary>
    public Dictionary<int, float> EstValueList { get; set; }
    /// <summary>
    /// Possible discrete actions of the node
    /// </summary>
    public int ActionSize { get; private set; }
    /// <summary>
    /// The node's children equal to the action size
    /// </summary>
    public TreeNode[] Children { get; private set; }
    /// <summary>
    /// A manifestation of the current observation state
    /// </summary>
    public Tensor Latent { get; private set; }
    /// <summary>
    /// The model's calculated policy for the childrens actions
    /// </summary>
    public float[] PolPred { get; private set; }
    /// <summary>
    /// Node's paternal/maternal parent
    /// </summary>
    public TreeNode Parent { get; private set; }
    /// <summary>
    /// Bootstrapped value for determining the PickAction
    /// </summary>
    public float AverageVal { get; private set; }
    /// <summary>
    /// Number of times this node has been visited in the MCTS search process
    /// </summary>
    public int NumVisits { get; set; }
    /// <summary>
    /// The estimated singular reward calculated by the model
    /// </summary>
    public float Reward { get; private set; }
    /// <summary>
    /// Used for calculating reduced influence of the ValuePrefix for deeper nodes
    /// </summary>
    public float Discount { get; private set; }
    /// <summary>
    /// Normalization process
    /// </summary>
    public MinMax MinMax { get; private set; }
    /// <summary>
    /// Returns if any children of the node have been visited
    /// </summary>
    public bool IsExpanded { get; set; }

    /// <summary>
    /// The estimated future rewards after this node
    /// </summary>
    public float ValuePrefix { get; private set; }
    /// <summary>
    /// Sum of rewards up to this node
    /// </summary>
    public float Value { get; set; }
    /// <summary>
    /// hidden reward values for the lstm
    /// </summary>
    public (Tensor, Tensor) HiddenReward { get; set; }
    /// <summary>
    /// Marker for the root node
    /// </summary>
    public bool IsRoot { get; set; }


    public TreeNode(
        int id,
        Tensor latent,
        int actionSize,
        float valPrefix,
        float value,
        float[] polPred,
        TreeNode parent = null,
        float reward = 0,
        Tensor hiddenReward_c = null,
         Tensor hiddenReward_h = null,
        float discount = 1,
        MinMax minmax = null,
        float targetPrefixValue = 0f,
        bool isRoot = false)
    {
        Id = id;
        ActionSize = actionSize;
        Children = new TreeNode[actionSize];
        Latent = latent;
        PolPred = polPred;
        ValuePrefix = valPrefix;
        Value = value;
        Parent = parent;
        AverageVal = 0;
        NumVisits = 0;
        Reward = reward;
        HiddenReward = (hiddenReward_c, hiddenReward_h);
        Discount = discount;
        MinMax = minmax;
        IsRoot = isRoot;

        EstValueList = new Dictionary<int, float>();
    }

    public void Insert(int nodeId, int actionIndex, Tensor latent, float valPrefix, float value, float[] polPred, float reward, (Tensor, Tensor) hiddenReward, MinMax minmax, float targetPrefixValue = 0f)
    {
        if (Children[actionIndex] == null)
        {
            TreeNode newChild = new TreeNode(
                nodeId,
                latent: latent,
                valPrefix: valPrefix,
                value: value,
                polPred: polPred,
                actionSize: ActionSize,
                parent: this,
                reward: reward,
                hiddenReward_c: hiddenReward.Item1,
                hiddenReward_h: hiddenReward.Item2,
                discount: Discount,
                minmax: minmax,
                targetPrefixValue: targetPrefixValue
            );

            Children[actionIndex] = newChild;
            IsExpanded = true;
        }
        else
        {
            throw new ArgumentException("This node has already been traversed");
        }
    }

    public void UpdateVal(float currentVal)
    {
        /// <summary>
        /// Updates the average value of a node when a new value is received.
        /// Copies the formula of the muzero paper rather than the neater form of
        /// just tracking the sum and dividing as needed.
        /// </summary>
        float numerator = AverageVal * NumVisits + currentVal;
        AverageVal = numerator / (NumVisits + 1);
    }



    private double GetQsa(int action)
    {
        TreeNode child = Children[action];
        System.Diagnostics.Debug.Assert(child.IsExpanded);
        double qsa = child.GetReward() + Discount * child.GetValue();
        return qsa;
    }

    public double GetVMix()
    {
        ///
        /// v_mix implementation, refer to https://openreview.net/pdf?id=bERaNdoegnO (Appendix D)
        /// 
        float[] policy = PolPred;// Softmax(PolPred); //PolPred is already softmaxed
        double piSum = 0;
        double piQsaSum = 0;

        for (int act = 0; act < Children.Length; act++)
        {
            TreeNode child = Children[act];
            if (child != null && child.IsExpanded)
            {
                piSum += policy[act];
                piQsaSum += policy[act] * GetQsa(act);
            }
        }

        double vMix = 0;
        // if no child has been visited
        if (piSum < .001)
            vMix = GetValue();
        else
        {
            int visitSum = Children.Sum(c => c == null ? 0 : c.NumVisits);
            vMix = (1d / (1d + visitSum)) * (GetValue() + visitSum * piQsaSum / piSum);
        }

        return vMix;
    }

    private double GetValue()
    {
        if (IsExpanded || Parent == null)
        {
            return AverageVal;
        }
        else
            return Parent.GetVMix();
    }

    public float GetReward()
    {
        if (Parent == null)
            return ValuePrefix;

        return ValuePrefix - Parent.ValuePrefix;
    }

    private double[] GetCompletedQ()
    {
        List<double> qList = new List<double>();

        double completedQ = 0;
        for (int act = 0; act < Children.Length; act++)
        {
            TreeNode child = Children[act];

            if (child != null && child.IsExpanded)
                completedQ = GetQsa(act);
            else
                completedQ = GetVMix();

            MinMax.Update(completedQ);
            qList.Add(completedQ);
        }

        for (int act = 0; act < Children.Length; act++)
        {
            qList[act] = MinMax.Normalize(qList[act]);
        }

        return qList.ToArray();
    }

    private double[] SigmaTransform(int maxChildVisitCount, double[] values, int cVisit = 50, float cScale = .1f)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = (cVisit + maxChildVisitCount) * cScale * values[i];
        return values;
    }

    public double[] GetTransformedCompletedQ()
    {
        int maxChildVisitCount = Children.Max(f => f == null ? 0 : f.NumVisits);
        double[] completedQ = GetCompletedQ();
        double[] transformedCompletedQ = SigmaTransform(maxChildVisitCount, completedQ);
        return transformedCompletedQ;
    }

    private float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] -= max;
            logits[i] = (float)Math.Exp(logits[i]);
        }

        float sum = logits.Sum();
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] /= sum;
        }

        return logits;
    }

    private float[] GetImprovedPolicy(double[] transCompletedQ)
    {
        float[] logits = new float[transCompletedQ.Length];
        for (int i = 0; i < Children.Length; i++)
        {
            logits[i] = (float)transCompletedQ[i] + PolPred[i];
        }

        //softmax function
        return Softmax(logits);
    }

    public double ActionScore(int actionIndex, int totalVisitCount)
    {
        /// <summary>
        /// Scoring function for the different potential actions, following the formula in Appendix B of muzero.
        /// </summary>
        double c1 = 1.25;
        double c2 = 19652;

        TreeNode child = Children[actionIndex];

        int visitCount = child != null ? child.NumVisits : 0;
        double qValue = child != null ? child.AverageVal : 0;

        // p here is the prior - the expectation of what the policy will look like
        float prior = PolPred[actionIndex];

        // This term increases the prior on those actions which have been taken only a small fraction
        // of the current number of visits to this node
        double exploreTerm = Math.Sqrt(totalVisitCount) / (1 + visitCount);

        // This is intended to more heavily weight the prior as we take more and more actions.
        double balanceTerm = c1 + Math.Log((totalVisitCount + c2 + 1) / c2);

        double score = MinMax.Normalize(qValue) + (prior * exploreTerm * balanceTerm);

        return score;
    }


    public int PickAction(bool isRoot)
    {
        /// <summary>
        /// Gets the score each of the potential actions and picks the one with the highest.
        /// </summary>
        int totalVisitCount = 0;
        foreach (var child in Children)
        {
            if (child != null)
            {
                totalVisitCount += child.NumVisits;
            }
        }

        // EfficientV2 method 
        // https://github.com/Shengjiewang-Jason/EfficientZeroV2/blob/main/ez/mcts/py_mcts.py#L609
        float[] improvedPolicy = GetImprovedPolicy(GetTransformedCompletedQ());
        float[] qScores = new float[Children.Length];
        for (int act = 0; act < Children.Length; act++)
        {
            TreeNode child = Children[act];
            if (child == null && isRoot)
                qScores[act] = 1; // ensures that every child of the root node is picked once
            else if(child == null)
                qScores[act] = improvedPolicy[act];
            else
                qScores[act] = improvedPolicy[act] - child.NumVisits / (1f + totalVisitCount);
        }
        int action = Utils.ArgMax(qScores);
        return action;


        double[] scores = new double[ActionSize];
        for (int i = 0; i < ActionSize; i++)
        {
            scores[i] = ActionScore(i, totalVisitCount);
        }
        double maxScore = scores[0];
        for (int i = 1; i < scores.Length; i++)
        {
            if (scores[i] > maxScore)
            {
                maxScore = scores[i];
            }
        }

        // Need to be careful not to always pick the first action if it is common that two are scored identically
        List<int> actionsList = new List<int>();
        for (int i = 0; i < ActionSize; i++)
        {
            if (scores[i] == maxScore)
            {
                actionsList.Add(i);
            }
        }

        if (actionsList.Count > 1)
        {
            Random random = new Random();
            int actionIdx = random.Next(0, actionsList.Count);
            return actionsList[actionIdx];
        }

        return actionsList[0];
    }

    public Act PickBestAction(double temperature)
    {
        /// <summary>
        /// Picks the action to actually be taken in game,
        /// taken by the root node after the full tree has been generated.
        /// Note that it only uses the visit counts, rather than the score or prior,
        /// these impact the decision only through their impact on where to visit.
        /// </summary>

        int[] visitCounts = new int[ActionSize];
        for (int i = 0; i < Children.Length; i++)
        {
            visitCounts[i] = Children[i] != null ? Children[i].NumVisits : 0;
        }

        int action;
        // zero temperature means always picking the the highest child values?? policy??
        if (temperature == 0)
        {
            float[] priors = new float[ActionSize];
            for (int i = 0; i < Children.Length; i++)
                priors[i] = Children[i].AverageVal;
            action = Utils.ArgMax(priors);
        }
        else
        {
            double[] actionProbs = new double[visitCounts.Length];
            double totalScore = 0;
            for (int i = 0; i < visitCounts.Length; i++)
            {
                actionProbs[i] = Math.Pow(visitCounts[i], 1 / temperature);
                totalScore += actionProbs[i];
            }
            for (int i = 0; i < actionProbs.Length; i++)
            {
                actionProbs[i] /= totalScore;
            }

            Random random = new Random();
            action = ChooseWithProbability(actionProbs, random);
        }

        // Prints a lot of useful information for how the algorithm is making decisions
        //if (Debug)
        //{
        //    double[] valPreds = new double[Children.Length];
        //    for (int i = 0; i < Children.Length; i++)
        //    {
        //        valPreds[i] = Children[i] != null ? (double)Children[i].ValPred : 0;
        //    }
        //    Console.WriteLine($"{string.Join(", ", visitCounts)}, {ValPred}, {string.Join(", ", valPreds)}, {(valPreds[0] > valPreds[1] ? "L" : "R")}, {(action == 0 && valPreds[0] > valPreds[1] || action == 1 && valPreds[0] <= valPreds[1] ? "T" : "F")}");
        //}

        return new Act(torch.tensor(action));
    }

    private int ChooseWithProbability(double[] probabilities, Random random)
    {
        double cumulative = 0.0;
        double randomValue = random.NextDouble();
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (randomValue < cumulative)
            {
                return i;
            }
        }
        return probabilities.Length - 1; // Return last index if not found in loop
    }
}
