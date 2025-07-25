﻿namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     This is demo net to guide how to create a new Module
    /// </summary>
    public sealed class Net : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Module<torch.Tensor, torch.Tensor> layers;

        public Net(long obsSize, long hiddenSize, long actionNum, DeviceType deviceType = DeviceType.CUDA)
            : base("Net")
        {
            var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("line1", Linear(obsSize, hiddenSize)),
                ("relu", ReLU()),
                ("line2", Linear(hiddenSize, actionNum))
            };
            layers = Sequential(modules);
            layers.to(new torch.Device(deviceType));
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            return layers.forward(input.to_type(torch.ScalarType.Float32));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                layers.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }
}