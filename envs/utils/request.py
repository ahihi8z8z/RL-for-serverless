import numpy as np

'''
Define resource usage for request type:
 request_type: [RAM, CPU, Power, Time]
    
'''    
class Request_usage:
    Type0 = np.array([10, 10, 10, 100])   
    Type1 = np.array([20, 20, 20, 200])   
    Type2 = np.array([30, 30, 30, 300])   
    Type3 = np.array([40, 40, 40, 400])   


class Request():
    def __init__(self, type: int, state: int = 0, 
                timeout: int = 0, in_queue_time: int = 0):
        self.type = type
        self.time_out =  timeout 
        self.in_queue_time = in_queue_time
        self.in_system_time = 0
        self.out_system_time = 0
        self.state = state 
        self.resource_usage = None
        self.active_time = 0
        self.set_resource_usage()

    def set_resource_usage(self):
        if 0 == self.type:
            self.active_time = Request_usage.Type0[3]
            self.resource_usage = Request_usage.Type0
        elif 1 == self.type:
            self.active_time = Request_usage.Type1[3]
            self.resource_usage = Request_usage.Type1
        elif 2 == self.type:
            self.active_time = Request_usage.Type2[3]
            self.resource_usage = Request_usage.Type2
        elif 3 == self.type:
            self.active_time = Request_usage.Type3[3]
            self.resource_usage = Request_usage.Type3
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

def generate_requests( current_time, size: int = 4, duration: int = 10, avg_requests_per_second: float = 2, timeout: int = 10):
    rng = np.random.default_rng()
    
    # Số lượng request trong khoảng thời gian duration (second)
    num_requests = rng.poisson(avg_requests_per_second * duration)
    
    # Thời gian giữa các request tuân theo phân phối mũ
    inter_arrival_times = rng.exponential(1.0 / avg_requests_per_second, num_requests)
    
    # Tính thời gian đến hệ thống của từng request
    arrival_times = np.cumsum(inter_arrival_times)
    
    requests = []
    for arrival_time in arrival_times:
        # Kiểm tra nếu thời gian đến vẫn nằm trong khoảng thời gian duration
        if arrival_time < duration :
            request = Request(type=rng.integers(0, size), in_queue_time=arrival_time+current_time, timeout=timeout)
            requests.append(request)
    return requests
