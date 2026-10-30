
import tensorflow as tf
import numpy as np
import warnings
import time 
from tqdm import tqdm
from gpshield.common.stats import Summary_Stats

class RolloutLogger:

    def __init__(self, log_every, stdout=True, tqdm=False, tensorboard=False, summary_writer=None, stats_window_size=100, prefix=''):
        self.stdout = stdout
        self.tqdm = tqdm
        self.tensorboard = tensorboard
        self.summary_writer = summary_writer
        if self.tensorboard:
            assert self.summary_writer is not None
        if (self.summary_writer is not None) and (not self.tensorboard):
            warnings.warn("tensorboard is set to False but summary writer is provided, this may produce unexpected behaviour")
        self.steps = 0
        self.log_every = log_every
        self.stats_window_size = stats_window_size
        self.prefix = prefix if prefix[-1] == '/' else prefix + '/'
        self.stats = {'ep_rew': [0.0],
                      'ep_len' : [0.0],
                      'is_success': [1.0],
                      'ep_overrides': [0.0]}
        self.start_time = None
    
    def reset(self):
        """this method simply resets the logging of the current episode"""
        """to completely reset the logger simply create a new instance"""
        for key in self.stats:
            if key in ['ep_rew', 'ep_len', 'ep_overrides']:
                self.stats[key][-1] = 0.0
            elif key in ['is_success']:
                self.stats[key][-1] = 1.0
            else:
                raise NotImplementedError(f'Logger stats key `{key}` not implemented')

    def step(self, info):
        if self.steps == 0:
            self.start_time = time.time()
        self.steps += 1
        assert 'done' in info
        for key in info:
            if key in ['ep_rew', 'ep_len', 'ep_overrides']:
                self.stats[key][-1] += info[key]
            elif key in ['is_success']:
                self.stats[key][-1] *= info[key]
            else:
                pass
        if info['done']:
            for key in self.stats:
                if key in ['ep_rew', 'ep_len', 'ep_overrides']:
                    self.stats[key].append(0.0)
                elif key in ['is_success']:
                    self.stats[key].append(1.0)
                else:
                    raise NotImplementedError(f'Logger stats key `{key}` not implemented')
        if (self.steps % self.log_every) == 0:
            self._log(self.steps)

    def _log(self, step):

        stats_to_log = {}
        

        for key, val in self.stats.items():
            if len(val) > 1:
                stats_to_log[key] = np.mean(val[:-1][-self.stats_window_size:])

        if self.start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                stats_to_log['fps'] = step / elapsed
    
        if self.tensorboard:
            with self.summary_writer.as_default():
                for key in stats_to_log:
                    tf.summary.scalar(self.prefix + key, stats_to_log[key], step=step)

        if self.stdout:
            stats_to_print = {key: "{0:.4g}".format(val) for key, val in stats_to_log.items()}
            if self.start_time is not None:
                stats_to_print['time_elapsed'] = "{0:.4g}".format(current_time - self.start_time)
            stats_to_print['total_timesteps'] = str(step)
            max_key_len = max([len(key) for key in stats_to_print] + [len(self.prefix) - 2])
            max_val_len = max([len(val) for val in stats_to_print.values()])
            stdout = ""
            max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
            stdout += ("-"*max_len + "\n")
            stdout += ("|  "+self.prefix + " "*(2+max_key_len-len(self.prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            for key, val in stats_to_print.items():
                stdout += ("|    "+key + " "*(max_key_len-len(key)+2) + "|  " + val  +" "*(max_val_len - len(val) + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
            if self.tqdm:
                tqdm.write(stdout)
            else:
                print(stdout)


class TrainLogger:

    def __init__(self, log_every, stdout=True, tqdm=False, tensorboard=False, summary_writer=None, stats_window_size=10, prefix=''):
        self.stdout = stdout
        self.tqdm = tqdm
        self.tensorboard = tensorboard
        self.summary_writer = summary_writer
        if self.tensorboard:
            assert self.summary_writer is not None
        if (self.summary_writer is not None) and (not self.tensorboard):
            warnings.warn("tensorboard is set to False but summary writer is provided, this may produce unexpected behaviour")
        self.steps = 0
        self.log_every = log_every
        self.stats_window_size = stats_window_size
        self.prefix = prefix if prefix[-1] == '/' else prefix + '/'
        self.stats = {}

    def reset(self):
        self.stats = {}

    def step(self, stats):
        self.steps += 1

        for key, val in stats.items():
            if isinstance(val, Summary_Stats):
                st = val.get()
                for k, v in st.items():
                    if k in self.stats:
                        self.stats[k].append(v)
                    else:
                        self.stats[k] = [v]
            else:
                if key in self.stats:
                    self.stats[key].append(val)
                else:
                    self.stats[key] = [val]

        if (self.steps % self.log_every) == 0:
            self._log(self.steps)

    def _log(self, step):

        stats_to_log = {}
        
        for key, val in self.stats.items():
            if len(val) > 0:
                stats_to_log[key] = np.mean(val[-self.stats_window_size:])
    
        if self.tensorboard:
            with self.summary_writer.as_default():
                for key in stats_to_log:
                    tf.summary.scalar(self.prefix + key, stats_to_log[key], step=step)

        if self.stdout:
            stats_to_print = {key: "{0:.4g}".format(val) for key, val in stats_to_log.items()}
            max_key_len = max([len(key) for key in stats_to_print] + [len(self.prefix) - 2])
            max_val_len = max([len(val) for val in stats_to_print.values()])
            stdout = ""
            max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
            stdout += ("-"*max_len + "\n")
            stdout += ("|  "+self.prefix + " "*(2+max_key_len-len(self.prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            for key, val in stats_to_print.items():
                stdout += ("|    "+key + " "*(max_key_len-len(key)+2) + "|  " + val  +" "*(max_val_len - len(val) + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
            if self.tqdm:
                tqdm.write(stdout)
            else:
                print(stdout)


        

    
        
