import os
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import utils.request as rq
# from utils.container import Container

'''
Define an index corresponding to the action that changes the container's state:
    - Destination state is changed from the original state: 1
    - Source state is changed to another state: -1
    - State is not changed: 0
'''
class Resource_Type:
    RAM = 0
    CPU = 1
    Power = 2
    Time = 3

'''    
Defines symbols in a state machine:
    - N  = Null
    - L0 = Cold
    - L1 = Warm Disk
    - L2 = Warm CPU
    - A  = Active
'''
class Container_States:
    Null = 0
    Cold = 1
    Warm_Disk = 2
    Warm_CPU = 3
    Active = 4
    
'''    
Defines request state:
'''
class Request_States:
    In_Queue = 0
    In_System = 1
    Time_Out = 2
    Done = 3

'''
Define resource usage of request:
 Request type: [RAM, CPU, Power, Time]  
'''      
Request_Resource_Usage = np.array([np.array([1, 1, 10]),   
                          np.array([2, 2, 20]),   
                          np.array([3, 3, 30]),   
                          np.array([4, 4, 40])])   
    
'''
Define cases where state changes can occur:
    N <-> L0 <-> L1 <-> L2 <-> A
'''    
Transitions = np.array([np.array([-1, 1, 0, 0, 0]),   # N -> L0
                        np.array([1, -1, 0, 0, 0]),   # L0 -> N
                        np.array([0, -1, 1, 0, 0]),   # L0 -> L1
                        np.array([0, 1, -1, 0, 0]),   # L1 -> L0
                        np.array([0, 0, -1, 1, 0]),   # L1 -> L2
                        np.array([0, 0, 1, -1, 0]),   # L2 -> L1
                        np.array([0, 0, 0, 0, 0])])   # No change
    
'''
Define transition cost for moving to another states:
 state: [RAM, CPU, Power, Time]  
'''    
Transitions_cost = np.array([np.array([0, 0, 50, 5]),      # N -> L0
                             np.array([0, 0, 50, 1.4]),    # L0 -> N
                             np.array([0, 0, 2000, 50]),   # L0 -> L1
                             np.array([0, 0, 50, 2]),      # L1 -> L0
                             np.array([0.9, 0, 100, 7]),   # L1 -> L2
                             np.array([0, 0, 400, 30]),    # L2 -> L1
                             np.array([1.2, 0, 0, 0])])    # No change 

'''
Define resource usage for staying in each state:
 state: [RAM, CPU, Power]
    
'''    
Container_Resource_Usage =np.array([np.array([0, 0, 0]),   # N
                          np.array([0, 0, 0]),             # L0
                          np.array([0, 0, 0]),             # L1
                          np.array([0.9, 0, 0]),           # L2
                          np.array([1.2, 0, 50])])         # A


class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=4):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.current_time = 0  # Start at time 0
        self.timestep = 60 
        self.size = size  # The number of services
        self.num_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (RAM, CPU, GPU)
        self.min_container = 16
        self.max_container = 256
        
        self.timeout = 10  # Set timeout value = 10s
        self.container_lifetime = 86400  # Set lifetime of a container = 1 day
        self.limited_resource = [1000,1000]  # Set limited amount of [RAM,CPU] of system
        self.limited_requests = 128  # Set the limit number of requests that can exist in the system = 1280
        self.average_requests = 5  # Set the average incoming requests per second = 5
        self.energy_cost = 0.00000463333 # unit cent/Jun 
        self.ram_profit = 0.00002632022 # unit cent/Gb
        self.cpu_profit = 0.00002632022 # unit cent/vcpu
             
        '''
        Initialize the state and other variables
        '''
        self.truncated = False
        self.terminated = False
        self.truncated_reason = ""
        self.temp_reward = 0 # Reward cache for each step
        self.current_resource_usage = np.zeros(self.num_resources)
        
        
        self._all_requests =  [[] for _ in range(self.size)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.size)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.size)] # Accepted requests in system until current time
        self._new_requests = [[] for _ in range(self.size)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.size)] # Timeout requests cache in a timestep
        
        self._action_matrix = None  # Set an initial value
        
        # Cache of container and state
        # TODO: thử random các trạng thái khởi tạo ma trận container 
        self._container_matrix = np.hstack((
            np.random.randint(self.min_container, self.max_container, size=(self.size, 1)),  # Initially the containers are in Null state
            np.zeros((self.size, self.num_states-1), dtype=np.int16)
        ))
        
        self.num_container = np.sum(self._container_matrix, axis=1) # số lượng container mỗi service
        
        # Observation matrices cache
        self._in_queue_requests_obs = np.zeros((self.size))
        self._in_system_requests_obs = np.zeros((self.size))
        self._timeout_requests_obs = np.zeros((self.size))
        self._done_requests_obs = np.zeros((self.size))

        '''
        Define observations (state space)
        '''
        self.observation_space = spaces.Dict({
            "in_queue_requests":  spaces.Box(low=0, high=self.limited_requests, shape=(self.size,1), dtype=np.int16),
            "in_system_requests":  spaces.Box(low=0, high=self.limited_requests, shape=(self.size,1), dtype=np.int16),
            "timeout_requests":  spaces.Box(low=0, high=self.limited_requests, shape=(self.size,1), dtype=np.int16),
            "done_requests":  spaces.Box(low=0, high=self.limited_requests, shape=(self.size,1), dtype=np.int16),
            "resource_usage": spaces.Box(low=-self.limited_resource[0], high=self.limited_resource[0], shape=(self.num_resources,1), dtype=np.int16),
            "container_matrix": spaces.Box(low=0, high=self.max_container, shape=(self.size, self.num_states), dtype=np.int16)
        })
        
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self.action_space = self._action_space_init() 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    # Tạo không gian action
    def _action_space_init(self):
        action_unit = spaces.MultiDiscrete(np.array([Transitions.shape[0]]*self.size), seed=42) # Hien tai co 7 transition * 4 service
        action_coefficient = spaces.MultiDiscrete(self.num_container) # Số lượng container tối đa được phép chuyển trạng thái mỗi service 
        return spaces.Tuple((action_coefficient, action_unit)) 

            
        
            
    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        '''
        for service in range(self.size):
            for request in self._in_queue_requests[service]:
                if  0 <= self.current_time - request.in_queue_time < self.timestep:
                    self._in_queue_requests_obs[service] += 1    
            for request in self._timeout_requests[service]:
                if 0 <= self.current_time - request.out_system_time < self.timestep:
                    self._timeout_requests_obs[service] += 1   
            for request in self._in_system_requests[service]:
                if 0 <= self.current_time - request.in_system_time < self.timestep:
                    self._in_system_requests_obs[service] += 1   
            for request in self._done_requests[service]:
                if 0 <= self.current_time - request.out_system_time < self.timestep:
                    self._done_requests_obs[service] += 1 

        return {
            "in_queue_requests":  self._in_queue_requests_obs,
            "in_system_requests":  self._in_system_requests_obs,
            "timeout_requests":  self._timeout_requests_obs,
            "done_requests":  self._done_requests_obs,
            "resource_usage": self.current_resource_usage,
            "container_matrix": self._container_matrix
        }

    def _get_info(self):
        '''
        Defines a function that returns system evaluation parameters
        '''
        
        return {
            "current_step": int(self.current_time/self.timestep)
        }

    def _get_reward(self):
        if self.truncated == True:
            return -10000
        else:
            return self.temp_reward
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.current_resource_usage.fill(0)

        # Reset giá trị của self._container_matrix
        self._container_matrix.fill(0)
        self._container_matrix[:, 0] = self.num_container
        
        
        self._all_requests =  [[] for _ in range(self.size)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.size)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.size)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.size)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.size)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.size)] # Timeout requests cache in a timestep
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
     
    def _cal_system_resource(self,action):
        # Tính tài nguyên tiêu thụ trên toàn hệ thống sau khi thực hiện action
        current_resource_usage = [0]*self.num_resources
        for resource in range(self.num_resources):
            for service in range(self.size):
                # Tài nguyên tiêu thụ tức thời bởi các request trong hệ thống
                for request in self._in_system_requests[service]:
                    current_resource_usage[resource] += request.resource_usage[resource]
                
                
                for state in range(self.num_states):
                    # Tài nguyên tiêu thụ tức thời tại các trạng thái của container
                    current_resource_usage[resource] +=  self._container_matrix[service][state]*Container_Resource_Usage[state][resource] 
                    
                    # Tài nguyên tiêu thụ tức thời do chuyển trạng thái
                    trans_type = action[1][service]
                    trans_num  = action[0][service]
                    # Kiểm tra xem chuyển trạng thái xong chưa
                    if self.current_time % self.timestep < Transitions_cost[trans_type][Resource_Type.Time]:
                        current_resource_usage[resource] += Transitions_cost[trans_type][resource]*trans_num
                
            self.current_resource_usage[resource] = current_resource_usage[resource] 
                    
    def _receive_new_requests(self):
        new_requets = rq.generate_requests(current_time=self.current_time, duration=self.timestep,avg_requests_per_second=self.average_requests)
        for request in new_requets:
            self._new_requests[request.type].append(request)
            self._all_requests[request.type].append(request)
            self._in_queue_requests[request.type].append(request)
        
        # Sắp xếp lại request trong queue theo thời gian đến hệ thống => FCFS
        for service in range(self.size):
            self._in_queue_requests[service].sort(key=lambda x: x.in_queue_time)
            
    def _set_truncated(self,action):
        if (np.any(self._container_matrix < 0)):
            self.truncated = True
            self.truncated_reason = "Wrong number action"
        
        # Dừng trainning episode nêú action làm tràn tài nguyên hệ thống
        # TODO: Phần này cần xem xét lại vì request đến cũng có thể là tràn tài nguyên hệ thống
        self._cal_system_resource(action)      
        if self.current_resource_usage[Resource_Type.RAM] > self.limited_resource[Resource_Type.RAM]:
            self.truncated = True
            self.truncated_reason = "RAM overloaded"
        if self.current_resource_usage[ Resource_Type.CPU] > self.limited_resource[Resource_Type.CPU]:
            self.truncated = True
            self.truncated_reason = "CPU overloaded"
            
    def _set_terminated(self):
        if (self.current_time >= self.container_lifetime):
            self.terminated = True            
    
    def _apply_action(self, action):
        self._action_to_matrix(action)
        self._container_matrix += self._action_matrix              
            
    def _handle_env_change(self,action):
        # Xử lý request mỗi giây 1 lần cho 1 timestep
        relative_time = 0
        while relative_time < self.timestep:
            for service in range(self.size):
                # Giảm active_time
                for request in self._in_system_requests[service]:
                    # Giải phóng các request đã thực hiện xong
                    if request.active_time == 0:
                        request.set_state(Request_States.Done)
                        request.set_out_system_time(self.current_time)
                        self._done_requests[service].append(request)
                        self._in_system_requests[service].remove(request)
                        self._container_matrix[service][Container_States.Active] -= 1
                        self._container_matrix[service][Container_States.Warm_CPU] += 1
                    else:
                        request.active_time -= 1
                
                # Giảm time out        
                for request in self._in_queue_requests[service]:
                    # Giải phóng các request bị time_out
                    if request.time_out == 0:
                        request.set_state(Request_States.Time_Out)
                        request.set_out_system_time(self.current_time)
                        self._timeout_requests[service].append(request)
                        self._in_queue_requests[service].remove(request)
                    elif request.in_system_time <= self.current_time:
                        request.tine_out -= 1
                        # Nếu còn tài nguyên trống, đẩy request vào 
                        if self._container_matrix[service][Container_States.Warm_CPU] > 0:
                            request.set_state(Request_States.In_System)
                            request.set_in_system_time(self.current_time)
                            self._in_system_requests[service].append(request)
                            self._in_queue_requests[service].remove(request)
                            self._container_matrix[service][Container_States.Active] -= 1
                            self._container_matrix[service][Container_States.Warm_CPU]    += 1
                
                # Tính năng lượng tiêu tốn trên toàn bộ hệ thống
                self._cal_system_resource(action)
                # Tính reward tức thời cho thời điểm hiện tại
                self._cal_temp_reward() 
                self.current_time += 1
                relative_time += 1
        
    def _cal_temp_reward(self):
        # TODO: đơn giản hóa reward, 
        # TODO: thêm based resource cho compute node chạy container để tránh tình trạng bật tất cả container warm cpu
        # TODO: nghĩ thêm về cách tính reward
        # TODO: phân tích các input có thể ảnh hướng đến kết quả
        abandone_penalty = 0
        delay_penalty = 0
        profit = 0
        energy_cost = self.current_resource_usage[Resource_Type.Power]*self.energy_cost
        
        for service in range(self.size):
            for request in self._in_system_requests[service]:
                # Delay penalty ở đây hiểu là tiền bị thiệt hại do request không được phục vụ ngay mà phải nằm trong queue
                # Delay penalty chỉ tính cho nhhững request được phục vụ bởi hệ thống, những request bị timeout sẽ tính vào abandone penalty
                # Delay penalty được một lần duy nhất tại thời điểm request được hệ thống accept
                if request.in_system_time == self.current_time:
                    delay_time = request.in_system_time - request.in_queue_time
                    delay_penalty += request.active_time*Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*(5+0.34*delay_time)/100
                    delay_penalty += request.active_time*Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit*(5+0.34*delay_time)/100
                # Profit của các request được chấp nhận vào hệ thống trong 1 giây 
                else:
                    profit += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit
                    profit += Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit
                
            for request in self._timeout_requests[service]:
                # Abandone penalty được một lần duy nhất tại thời điểm request hết timeout và bị hệ thống reject
                if request.out_system_time == self.current_time:
                    in_queue_time = request.out_system_time - request.in_system_time
                    abandone_penalty += request.active_time*Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*(5.8*in_queue_time)/100
                    abandone_penalty += request.active_time*Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*(5.8*in_queue_time)/100
            
        self.temp_reward += (profit - delay_penalty - abandone_penalty - energy_cost) 
    
    def _action_to_matrix(self,action):
        action_coefficient = np.diag(action[0])
        action_unit = []
        for service in action[1]:
            action_unit.append(Transitions[service])
        self._action_matrix = action_coefficient @ action_unit
        
    def _clear_cache(self):
        self._new_requests = [[] for _ in range(self.size)] 
        
        # Observation matrices cache
        self._in_queue_requests_obs.fill(0)
        self._in_system_requests_obs.fill(0)
        self._timeout_requests_obs.fill(0)
        self._done_requests_obs.fill(0)
        
        self.temp_reward = 0
        self.truncated = False
        self.terminated = False
                 
       
    def _pre_step(self,action):
        self._clear_cache()
        self._apply_action(action)     # Thay đổi ma trận trạng thái của container
        self._set_terminated()
        self._set_truncated(action)
        
        
    def step(self, action):
        self._pre_step(action)
        
        self._receive_new_requests()   # Nhận request đến hệ thống trong timestep
        self._handle_env_change(action)      
        observation = self._get_obs()
        info = self._get_info() 
        reward = self._get_reward()
        
        return observation, reward, self.terminated, self.truncated, info
    
    def render(self):
        '''
        Implement a visualization method
        '''
        obs = self._get_obs()
        reward = self._get_reward()
        print("SYSTEM EVALUATION PARAMETERS:")
        print("- Rewards: {:.2f}".format(reward))
        print("- Energy Consumption : {:.2f}J".format(obs["resource_usage"][Resource_Type.Power]))
        print("- RAM Consumption  : {:.2f}Gb".format(obs["resource_usage"][Resource_Type.RAM]))
        print("- CPU Consumption  : {:.2f}Core".format(obs["resource_usage"][Resource_Type.CPU]))

    def close(self):
        '''
        Implement the close function to clean up (if needed)
        '''
        pass


if __name__ == "__main__":
    # Create the serverless environment
    env = ServerlessEnv()
    
    # Reset the environment to the initial state
    observation, info = env.reset()
       
    # Perform random actions
    i = 0
    while (True):
        i += 1
        print("----------------------------------------")
        action = None
        while (True):
            action = env.action_space.sample()  # Random action
            # print("Action:\n", action)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated == False:
                break
            else:
                env.reset()
        print(f"Round: {i}, Done: {terminated}")    
        env._cal_system_resource(action)
        env.render()    
        if (terminated): 
            print("----------------------------------------")
            break
        else: continue