using DeepSharp.RL.Environs;
using MuZero.Sharp;

namespace LunarLander;

public class SPW : MCTS
{
    private float _alpha;

    /// <summary>
    /// Simple Progressive Widening trees based on Monte Carlo Tree Search for Continuous and
    /// Stochastic Sequential Decision Making Problems, Courtoux
    /// </summary>
    /// <param name="alpha">the number of children of a decision node are always greater that v**alpha,
    /// where v is the number of visits to the current decision node</param>
    /// <param name="initObs">initial state of the tree. Returned by env.reset()</param>
    /// <param name="env">game environment</param>
    /// <param name="k">exploration parameter of UCB</param>
    public SPW(float alpha, Observation initObs, Environ<Space, Space> env, float k)
        : base(initObs, env, k)
    {
        _alpha = alpha;
    }

    /// <summary>
    ///  Selects the action to play from the current decision node. The number of children of a DecisionNode is
    /// kept finite at all times and monotonic to the number of visits of the DecisionNode.
    /// </summary>
    /// <param name="node">current decision node</param>
    /// <returns>action to play</returns>
    public override Act SelectAction(INode node)
    {
        if (Math.Pow(node.Visits, _alpha) >= node.Children.Count)
            return Env.SampleAct();
        else
        {
            foreach (var child in node.Children.Values)
            {
                if (child.Visits > 0)
                    child.UtcScore = ComputeUTCB(node, child, K);
                else
                    child.UtcScore = float.MinValue;
            }

            INode maxNode = node.Children.Values.OrderByDescending(f => f.UtcScore).First();

            return maxNode.Action;
        }
    }

    private float ComputeUTCB(INode node, INode childNode, float k)
    {
        return (float)((childNode.CumulativeReward / childNode.Visits) + (K * Math.Sqrt(Math.Log(node.Visits) / childNode.Visits)));

    }
}
