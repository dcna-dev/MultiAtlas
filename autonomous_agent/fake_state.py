import random


class FakeState:
    def __init__(self):
        pass

    def get_metric(self, metric=0, action=3):
        
        if metric == 0:
            return round(random.random(), 2)
        
        if action == 0:
            return round(random.uniform(0.0, metric), 2)
        if action == 1:
            return round(random.uniform(metric, 1.0), 2)
        if action == 2:
            return metric
    
    def get_state(self, state, action):
        #import pdb; pdb.set_trace()
        return [self.get_metric(metric=metric, action=action) for metric in state]