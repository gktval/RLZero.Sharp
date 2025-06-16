using DeepSharp.RL.Environs;
using DeepSharp.RL.ExperienceSources;

namespace DeepSharp.RL.ExpReplays
{
    /// <summary>
    ///     Exp Relay apply for Store steps
    /// </summary>
    public abstract class ExpReplay
    {
        private readonly torch.Device _device;
        protected ExpReplay(torch.Device device, int capacity = 10000)
        {
            Capacity = capacity;
            Buffers = new Queue<Step>(capacity);
            _device = device;
        }

        /// <summary>
        ///     Capacity of Experience Replay Buffer
        /// </summary>
        public int Capacity { protected set; get; }

        /// <summary>
        ///     Cache
        /// </summary>
        public Queue<Step> Buffers { set; get; }

        public int Size => Buffers.Count();

        /// <summary>
        ///     Record a step [State, Action, Reward, NewState]
        /// </summary>
        /// <param name="step"></param>
        public virtual void Enqueue(Step step)
        {
            if (Buffers.Count == Capacity) Buffers.Dequeue();
            Buffers.Enqueue(step);
        }

        /// <summary>
        ///     Record steps {[State , Action, Reward, NewState],...,[State , Action, Reward, NewState]}
        /// </summary>
        public void Enqueue(IEnumerable<Step> steps)
        {
            steps.ToList().ForEach(Enqueue);
        }

        protected abstract Step[] SampleSteps(int batchsize);

        public virtual void Enqueue(Episode episode)
        {
            Enqueue(episode.Steps);
        }

        public virtual ExperienceCase Sample(int batchsize)
        {
            var batchStep = SampleSteps(batchsize);

            /// Get Array from Steps
            var stateArray = batchStep.Select(a => a.PreState.Value!.unsqueeze(0).to(_device)).ToArray();
            var actArray = batchStep.Select(a => a.Action.Value!.unsqueeze(0).to(_device)).ToArray();
            var rewardArray = batchStep.Select(a => a.Reward.Value).ToArray();
            var stateNextArray = batchStep.Select(a => a.PostState.Value!.unsqueeze(0)).ToArray();
            var doneArray = batchStep.Select(a => a.IsComplete).ToArray();

            /// Convert to VStack
            var state = torch.vstack(stateArray);
            var actionV = torch.vstack(actArray).to(torch.ScalarType.Int64);
            var reward = torch.from_array(rewardArray).reshape(batchsize).to(_device);
            var stateNext = torch.vstack(stateNextArray);
            var done = torch.from_array(doneArray).reshape(batchsize).to(_device);

            var excase = new ExperienceCase(state, actionV, reward, stateNext, done);
            return excase;
        }


        public void Clear()
        {
            Buffers.Clear();
        }
    }
}