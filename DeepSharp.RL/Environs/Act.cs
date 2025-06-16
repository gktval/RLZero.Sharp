using System.Diagnostics.CodeAnalysis;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     动作
    /// </summary>
    public class Act 
    {
        public Act(torch.Tensor? action)
        {
            Value = action;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        /// The Action Value 
        /// </summary>
        public torch.Tensor? Value { set; get; }

        /// <summary>
        /// Identifying timestamp
        /// </summary>
        public DateTime TimeStamp { set; get; }


        //public bool Equals(Act? x, Act? y)
        //{
        //    if (ReferenceEquals(x, y)) return true;
        //    if (ReferenceEquals(x, null)) return false;
        //    if (ReferenceEquals(y, null)) return false;
        //    return x.GetType() == y.GetType() && x.Value!.Equals(y.Value!);
        //}

        //public int GetHashCode(Act obj)
        //{
        //    return HashCode.Combine(obj.TimeStamp, obj.Value);
        //}

        public Act To(torch.Device device)
        {
            return new Act(Value!.to(device));
        }

        public override string ToString()
        {
            return $"{Value!.ToString(torch.numpy)}";
        }

        public override bool Equals(object? obj)
        {
            Act act = obj as Act;
            if (act != null)
            {
                return act.ToString() == this.ToString();
            }
            return false;
        }

        public override int GetHashCode()
        {
            return this.ToString().GetHashCode();
        }
        //public bool Equals(Act? x, Act? y)
        //{
        //    if (x != null && y != null)
        //    {
        //        return x.ToString() == y.ToString();
        //    }
        //    return false;
        //}

        //public int GetHashCode(Act obj)
        //{
        //    return obj.ToString().GetHashCode();
        //}
    }
}