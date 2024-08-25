import numpy as np
import math
import uuid

'''
Define resource usage for request type:
 request_type: [RAM, CPU, Power]
    
'''    
Request_usage = np.array([np.array([10, 10, 10]),
                          np.array([20, 20, 20]),
                          np.array([30, 30, 30]),
                          np.array([40, 40, 40])])

Request_active_time = np.array([240, 200, 300, 400])

def ran_norm_gen(mean, std_dev):
    # Generate a random value following normal distribution
    value = np.random.normal(loc=mean, scale=std_dev)
    # Round to the nearest integer
    int_value = round(value)
    # Ensure the value is greater than 0
    positive_int_value = max(1, int_value)  # Ensures the value is at least 1
    return positive_int_value

class Request():
    def __init__(self, type: int, state: int = 0, 
                timeout: int = 0, in_queue_time: int = 0, active_time: int = 0):
        self._uuid = uuid.uuid1()
        self.type = type
        self.time_out =  timeout 
        self.in_queue_time = in_queue_time
        self.in_system_time = 0
        self.out_system_time = 0
        self.state = state 
        self.resource_usage = None
        self.active_time = active_time
        self.set_resource_usage()

    def set_resource_usage(self):
        self.resource_usage = Request_usage[self.type]
        
    def set_active_time(self, a):
        self.active_time = a
        
    def set_time_out(self, a):
        self.time_out = a
        
    def set_in_queue_time(self, a):
        self.in_queue_time = a if a >= 0 else 0
        
    def set_in_system_time(self, a):
        self.in_system_time = a
        
    def set_out_system_time(self, a):
        self.out_system_time = a
    
    def set_state(self, state):
        self.state = state

def generate_requests( current_time, size: int = 4, duration: int = 10, avg_requests_per_second: float = 2, timeout: int = 10, max_rq_active_time: int = 240):
    rng = np.random.default_rng()
    num_requests = rng.poisson(avg_requests_per_second * duration)
    inter_arrival_times = rng.exponential(1.0 / avg_requests_per_second, num_requests)
    arrival_times = np.cumsum(inter_arrival_times)
    requests = []
    
    for arrival_time in arrival_times:
        # Kiểm tra nếu thời gian đến vẫn nằm trong khoảng thời gian duration
        if arrival_time < duration :
            type = rng.integers(0, size)
            
            if max_rq_active_time["type"] == "random":
                active_time = ran_norm_gen(max_rq_active_time["value"], max_rq_active_time["value"]/10)
            else:
                active_time = max_rq_active_time["value"] if max_rq_active_time["value"] else Request_active_time[type]
            
            request = Request(type=type, in_queue_time=int(arrival_time+current_time), timeout=timeout, active_time=active_time)    
            requests.append(request)
    return requests
