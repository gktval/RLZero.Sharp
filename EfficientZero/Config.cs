using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EfficientZero;

public class Config
{
    // Setup params  
    public int seed { get; set; }
    public string log_dir { get; set; }
    public string log_name { get; set; }
    public bool load_buffer { get; set; }
    public bool debug { get; set; }
    public bool train_speed_profiling { get; set; }
    public bool get_batch_profiling { get; set; }
    public bool render { get; set; }
    public bool print_simple { get; set; }
    public int max_games { get; set; } // Total number of games before training is ended
    public int max_total_frames { get; set; }
    public int max_frames { get; set; }// Maximum frames for a single game before it is cut short
    public int[] obs_shape { get; set; } 
    public bool image { get; set; }


    // Model params
    public int latent_size { get; set; }
    public int act_units { get; set; }
    public int dyn_units { get; set; }
    public int fc_units { get; set; }
    public int support_width { get; set; }
    public int n_simulations { get; set; }
    public bool downsample { get; set; }
    public bool init_zero { get; set; }
    public int num_blocks { get; set; }
    public int num_channels { get; set; }
    public int reduced_channels { get; set; }
    public bool action_embedding { get; set; }
    public int action_embedding_dim { get; set; }

    public int[] projection_layers { get; set; } // hidden dim, output dim
    public int[] prjection_head_layers { get; set; }  // hidden dim, output dim


    // Training params
    public float learning_rate { get; set; }
    public float learning_rate_decay { get; set; }
    public float weight_decay { get; set; }
    public float grad_clip { get; set; }
    public float val_weight { get; set; }
    public int batch_size { get; set; }


    // Search params
    public float root_dirichlet_alpha { get; set; }
    public float explore_frac { get; set; }
    public float discount { get; set; }
    public int n_batches { get; set; }
    public int rollout_depth { get; set; }
    public int reward_depth { get; set; }
    public int buffer_size { get; set; }


    // Priority replay params
    public bool priority_replay { get; set; }
    public float priority_alpha { get; set; }
    public float priority_beta { get; set; }


    // Temperature schedule
    public int temp1 { get; set; } // Steps after which temperature is dropped to 0.5
    public int temp2 { get; set; } // Steps after which temperature is dropped to 0.25
    public int temp3 { get; set; } // Steps after which temperature is dropped to 0


    // Reanalyse
    public bool reanalyze { get; set; }
    public int reanalyse_n { get; set; }
    public float prior_weight { get; set; }
    public float momentum { get; set; }


    //EfficientZero Additions
    // Value prefix
    public bool value_prefix { get; set; }
    public int lstm_hidden_size { get; set; }


    // Off policy correction
    public bool off_policy_correction { get; set; }
    public float tau { get; set; }
    public int reward_steps { get; set; }
    public int total_training_steps { get; set; }


    // Consistency loss
    public bool consistency_loss { get; set; }
    public float consistency_weight { get; set; }

}
