namespace EfficientZero;

public static class Utils
{
    private static Random _random = new Random();

    public static int GetRandomFromProb(int[] indices, float[] probs)
    {
        var p = _random.NextDouble();
        float probSum = 0;
        int selectedIndex = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            probSum += probs[i];
            if (p <= probSum)
            {
                selectedIndex = indices[i];
                break;
            }
        }

        return selectedIndex;
    }

    public static int ArgMax(float[] args)
    {
        float max = args[0];
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] > max)
                max = args[i];
        }

        List<int> maxlist = new List<int>();
        for (int i = 0; i < args.Length; i++)
        {
            if (max == args[i])
                maxlist.Add(i);
        }

        if (maxlist.Count > 1)
        {
            int r = _random.Next(0, maxlist.Count);
            return maxlist[r];
        }

        return maxlist[0];
    }

    public static int[] GetRandomFromProb(int[] indices, float[] probs, int count)
    {
        List<double> pList = new List<double>();
        for (int i = 0; i < count; i++)
            pList.Add(_random.NextDouble());

        pList.Sort();

        float testSum = probs.Sum();
        for (int i = 0; i < probs.Length; i++)
            probs[i] = probs[i] / testSum;

        float probSum = 0;
        List<int> selectedIndices = new List<int>();
        int pIndex = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            probSum += probs[i];
            if (pList[pIndex] <= probSum)
            {
                selectedIndices.Add(indices[i]);
                pIndex += 1;
            }

            if (pIndex >= count)
                break;
        }

        return selectedIndices.ToArray();
    }

    internal static int[] GetIndices(int totalVals, int batchSize)
    {
        List<int> indices = new List<int>();
        for (int i = 0; i < totalVals; i++)
            indices.Add(i);

        Shuffle(indices);

        return indices.Take(batchSize).ToArray();
    }

    internal static int[] GetIndices(int totalVals, int batchSize, float[] probs)
    {
        List<int> indices = new List<int>();
        for (int i = 0; i < totalVals; i++)
            indices.Add(i);

        return GetRandomFromProb(indices.ToArray(), probs, batchSize);
    }

    public static void Shuffle<T>(this IList<T> list)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = _random.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }

    public static float[] AddDirichlet(float[] prior, double dirichletAlpha, double exploreFraction)
    {
        float[] newPrior = new float[prior.Length];
        var noise = new MathNet.Numerics.Distributions.Dirichlet(dirichletAlpha, 1);
        for (int i = 0; i < prior.Length; i++)
            newPrior[i] = (float)((1 - exploreFraction) * prior[i] + exploreFraction * noise.RandomSource.NextDouble());
        return newPrior;
    }
}
