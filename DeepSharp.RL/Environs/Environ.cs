using System.Text;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     环境
    ///     提供观察 并给与奖励
    /// </summary>
    public abstract class Environ<T1, T2>
        where T1 : Space
        where T2 : Space
    {
        public Action<Step>? CallBack;

        protected Environ(string name, DeviceType deviceType = DeviceType.CUDA)
        {
            Name = name;
            Device = new torch.Device(deviceType);
            Reward = new Reward(0);
            ObservationList = new List<Observation>();
        }

        protected Environ(string name)
            : this(name, DeviceType.CUDA)
        {
        }


        public string Name { set; get; }

        public torch.Device Device { set; get; }
        public T1? ActionSpace { protected set; get; }
        public T2? ObservationSpace { protected set; get; }
        public int[] ObservationDim { protected set; get; }
        public int Epoch => ObservationList.Count;
        /// <summary>
        /// Tells the underlying environment if the action should be rendered
        /// </summary>
        public bool Render { get; set; }

        /// <summary>
        ///     Observation Current
        /// </summary>
        public Observation? Observation { set; get; }

        /// <summary>
        ///     Reward Current
        /// </summary>
        public Reward Reward { set; get; }

        /// <summary>
        ///     Observation Temp List
        /// </summary>
        public List<Observation> ObservationList { set; get; }

        /// <summary>
        /// Resets the Environment 
        /// </summary>
        public virtual Observation Reset()
        {
            Observation = new Observation(ObservationSpace!.Generate());
            ObservationList = new List<Observation> {Observation};
            Reward = new Reward(0);
            return Observation;
        }

        /// <summary>
        ///     获取一个回合的最终奖励
        /// </summary>
        /// <param name="episode"></param>
        /// <returns></returns>
        public abstract float GetReturn(Episode episode);

        public virtual Act SampleAct()
        {
            return new Act(ActionSpace!.Sample());
        }

        ///// <summary>
        ///// Agent provide Act
        ///// </summary>
        ///// <param name="act"></param>
        ///// <returns></returns>
        //public virtual Step Step(Act act, int epoch)
        //{
        //    var state = Observation!;
        //    var stateNew = Update(act);
        //    var reward = GetReward(stateNew);
        //    var complete = IsComplete(epoch);
        //    var step = new Step(state, act, stateNew, reward, complete);
        //    ObservationList.Add(stateNew);
        //    Observation = stateNew;
        //    return step;
        //}

        /// <summary>
        /// Agent provide Act
        /// </summary>
        /// <param name="act"></param>
        /// <returns></returns>
        public virtual Step Step(Act act, int epoch)
        {
            var state = Observation!;
            var stateNew = Update(act);
            var reward = GetReward(stateNew);
            var complete = IsComplete(epoch);
            var step = new Step(state, act, stateNew, reward, complete);
            ObservationList.Add(stateNew);
            Observation = stateNew;
            return step;
        }


        /// <summary>
        /// Update Environ Observation according  with one action from Agent
        /// </summary>
        /// <param name="act">Action from Policy</param>
        /// <returns>new observation</returns>
        public abstract Observation Update(Act act);


        /// <summary>
        ///     Cal Reward from Observation
        ///     从观察获取单步奖励的计算方法
        /// </summary>
        /// <param name="observation">one observation</param>
        /// <returns>one reward</returns>
        public abstract Reward GetReward(Observation observation);


        ///// <summary>
        ///// Check Environ is Complete
        ///// </summary>
        ///// <param name="epoch"></param>
        ///// <returns></returns>
        //public abstract bool IsComplete(int epoch);

        /// <summary>
        /// Check Environ is Complete
        /// </summary>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public abstract bool IsComplete(int epoch);


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"{Name}\tLife:{ObservationList.Count}");
            str.AppendLine(new string('-', 30));
            str.Append($"State:\t{Observation!.Value!.ToString(torch.numpy)}");
            return str.ToString();
        }

        public virtual Environ<Space,Space> Clone()
        {
            throw new NotImplementedException();
        }

        public virtual void Close()
        {
            //overridden in inherited class
        }
    }
}