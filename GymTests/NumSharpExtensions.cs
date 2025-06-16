using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LunarLander;

public static class NumSharpExtensions
{
    public static float[] ToFloatArray(this NDArray npArray)
    {
        try
        {
            float[] doubleArray = npArray.ToArray<float>();
            return Array.ConvertAll(doubleArray, item => (float)item);
        }
        catch (Exception)
        {

            double[] doubleArray = npArray.ToArray<double>();
            return Array.ConvertAll(doubleArray, item => (float)item);
        }
    }

    public static float[,] ToFloat2DArray(this NDArray ndArray)
    {
        var multidimArray = ndArray.ToMuliDimArray<float>();
        var floatArray = new float[multidimArray.GetLength(0), multidimArray.GetLength(1)];

        for (int i = 0; i < multidimArray.GetLength(0); i++)
        {
            for (int j = 0; j < multidimArray.GetLength(1); j++)
            {
                floatArray[i, j] = (float)multidimArray.GetValue(i, j);
            }
        }

        return floatArray;
    }

    public static float[,,] ToFloat3DArray(this NDArray ndArray)
    {
        var multidimArray = ndArray.ToMuliDimArray<float>();
        var floatArray = new float[multidimArray.GetLength(2), multidimArray.GetLength(0), multidimArray.GetLength(1)];

        for (int i = 0; i < multidimArray.GetLength(0); i++)
        {
            for (int j = 0; j < multidimArray.GetLength(1); j++)
            {
                for (int k = 0; k < multidimArray.GetLength(2); k++)
                {
                    floatArray[k, i, j] = (float)multidimArray.GetValue(i, j, k);
                }
            }
        }

        return floatArray;
    }

    public static float[][][] ToJagged3DArray(this NDArray ndArray)
    {
        var multidimArray = ndArray.ToMuliDimArray<float>();
        var floatArray = new float[multidimArray.GetLength(0)][][];

        for (int i = 0; i < multidimArray.GetLength(0); i++)
        {
            floatArray[i] = new float[multidimArray.GetLength(1)][];
            for (int j = 0; j < multidimArray.GetLength(1); j++)
            {
                floatArray[i][j] = new float[multidimArray.GetLength(2)];
                for (int k = 0; k < multidimArray.GetLength(2); k++)
                {
                    floatArray[i][ j][ k] = (float)multidimArray.GetValue(i, j, k);
                }
            }
        }

        return floatArray;
    }
}
