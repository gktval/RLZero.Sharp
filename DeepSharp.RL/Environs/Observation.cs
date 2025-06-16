using static System.Runtime.InteropServices.JavaScript.JSType;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    /// The Observation State
    /// </summary>
    public class Observation: IComparable<Observation>
    {
        public Observation(torch.Tensor? state)
        {
            Value = state;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        /// Tensor Value
        /// </summary>
        public torch.Tensor? Value { set; get; }

        /// <summary>
        /// Value for keeping track of the hash
        /// </summary>
        public DateTime TimeStamp { set; get; }

        public Observation To(torch.Device device)
        {
            return new Observation(Value?.to(device));
        }

        public object Clone()
        {
            return new Observation(Value) {TimeStamp = TimeStamp};
        }

        public override string ToString()
        {
            return $"Observation\r\n{Value?.ToString(torch.numpy)}";
        }

        public int CompareTo(Observation? other)
        {
            return TimeStamp.CompareTo(other!.TimeStamp);
        }

        public override int GetHashCode()
        {
            DateTime origin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            TimeSpan diff = TimeStamp - origin;
            return (int)Math.Floor(diff.TotalSeconds);
        }

        public override bool Equals(object? obj)
        {
            if (obj is null || obj.GetType() != typeof(Observation))
                return false;

            Observation space = (Observation)obj;
            return space.TimeStamp.Ticks == TimeStamp.Ticks;
        }
    }
}