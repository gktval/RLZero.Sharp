﻿using FluentAssertions;
using MathNet.Numerics.Random;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    /// Action Space and Observation Space Parent Class
    /// </summary>
    public abstract class Space : IDisposable
    {
        protected Space(
            long[] shape,
            torch.ScalarType type,
            DeviceType deviceType,
            long seed)
        {
            (Shape, Type, DeviceType) = (shape, type, deviceType);
            CheckInitParameter(shape, type);
            CheckType();
            Generator = torch.random.manual_seed(new SystemRandomSource().NextInt64(0, 1000));
            N = shape.Aggregate(1, (a, b) => (int) (a * b));
        }

        public int N { get; init; }
        public long[] Shape { get; }
        public torch.ScalarType Type { get; }
        public DeviceType DeviceType { get; }
        internal torch.Generator Generator { get; }
        internal torch.Device Device => new(DeviceType);

        public void Dispose()
        {
            Generator.Dispose();
        }

        /// <summary>
        /// Returns a sample from the space.
        /// </summary>
        /// <returns></returns>
        public abstract torch.Tensor Sample();

        public abstract void CheckType();

        /// <summary>
        /// Generates a tensor whose shape and type are consistent with the space definition.
        /// </summary>
        /// <returns></returns>
        public virtual torch.Tensor Generate()
        {
            return torch.zeros(Shape, Type, Device);
        }


        public override string ToString()
        {
            return $"Space Type: {GetType().Name}\nShape: {Shape}\ndType: {Type} \nN:{N}";
        }

        private static void CheckInitParameter(long[] shape, torch.ScalarType type)
        {
            shape.Should().NotBeNull();
            shape.Length.Should().BeGreaterThan(0);
        }
    }
}