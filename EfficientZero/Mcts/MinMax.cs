using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EfficientZero;

public class MinMax
{
    /*
    This class tracks the smallest and largest values that have been seen
    so that it can normalize the values
    this is for when deciding which branch of the tree to explore
    by putting the values on a 0-1 scale, they become comparable with the probabilities
    given by the prior

    It comes pretty much straight from the MuZero pseudocode
    */

    private double maxValue;
    private double minValue;

    public MinMax()
    {
        // initialize at +-inf so that any value will supercede the max/min
        maxValue = double.NegativeInfinity;
        minValue = double.PositiveInfinity;
    }

    public void Update(double value)
    {
        maxValue = Math.Max(value, maxValue);
        minValue = Math.Min(value, minValue);
    }

    public double Normalize(double value)
    {
        // places value between 0 - 1 linearly depending on where it sits between minValue and maxValue
        if (maxValue > minValue)
        {
            if (value > maxValue)
                value = maxValue;
            else if (value < minValue)
                value = minValue;

            // We normalize only when we have set the maximum and minimum values.
            return (value - minValue) / Math.Max(maxValue - minValue, .01d);
        }
        else
        {
            return Math.Max(Math.Min(value, 1), 0);
        }
    }
}
