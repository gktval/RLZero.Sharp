using DeepSharp.RL.Environs;
using MathNet.Numerics;
using MuZero.Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace LunarLander;

public class DPW : SPW
{
    private float _alpha;
    private float _beta;
    private Random _random;

    /// <summary>
    /// Double Progressive Widening trees based on MCTS for Continuous and
    /// Stochastic Sequential Decision Making Problems, Courtoux.
    /// </summary>
    /// <param name="alpha">the number of children of a decision node are always greater that v**alpha,
    /// where v is the number of visits to the current decision node</param>
    /// <param name="beta">the number of outcomes of a random node is grater that v**beta,
    /// where v is the number of visits of the random node</param>
    /// <param name="initObs">initial state of the tree. Returned by env.reset().</param>
    /// <param name="env">game environment</param>
    /// <param name="k">exploration parameter of UCB</param>
    public DPW(float alpha, float beta, Observation initObs, Environ<Space, Space> env, float k)
        : base(alpha, initObs, env, k)
    {
        _alpha = alpha;
        _beta = beta;
        _random = new Random();
    }

    /// <summary>
    /// The number of outcomes of a RandomNode is kept fixed at all times and increasing
    /// in the number of visits of the random_node
    /// </summary>
    /// <param name="env">Clone of the current environment</param>
    /// <param name="randNode">random node from which to select the next state</param>
    /// <returns></returns>
    public override INode SelectOutcome(Environ<Space, Space> env, INode randNode)
    {
        if (Math.Pow(randNode.Visits, _beta) >= randNode.Children.Count)
        {
            var obs = env.Update(randNode.Action);
            var outcomeNode = new DecisionNode(obs, randNode, isFinal: env.IsComplete(1)); //IsComplete(1) is set to 1 so that the outcome is never forced to IsDone
            outcomeNode.Reward = env.GetReward(obs).Value;
            outcomeNode.Visits = randNode.Visits;
            outcomeNode.CumulativeReward = randNode.CumulativeReward;
            outcomeNode.Action = randNode.Action;
            return outcomeNode;
        }
        else
        {
            var unnormProbs = randNode.Children.Values.Select(f => f.Visits).ToList();
            List<float> probs = new List<float>();
            float sum = unnormProbs.Sum();
            for (int i = 0; i < unnormProbs.Count; i++)
            {
                probs.Add(unnormProbs[i] / sum);
            }

            var p = _random.NextDouble();
            float probSum = 0;
            INode selectedNode = null;
            for (int i = 0; i < probs.Count; i++)
            {
                probSum += probs[i];
                if (p <= probSum)
                {
                    selectedNode = randNode.Children.ElementAt(i).Value;
                    break;
                }
            }

            return selectedNode;
        }
    }
}
