import numpy as np

import gymnasium as gym
from gymnasium import spaces

import utils.request as rq
import itertools
import math

def compute_formula(num_box, num_ball):
    numerator = math.factorial(num_box + num_ball - 1)
    denominator = math.factorial(num_box - 1) * math.factorial(num_ball)
    return int(numerator/denominator)

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
Transitions_cost = np.array([np.array([1.2, 0, 0, 0]),# No change 
                             np.array([0, 0, 50, 5]),      # N -> L0
                             np.array([0, 0, 50, 1.4]),    # L0 -> N
                             np.array([0, 0, 2000, 50]),   # L0 -> L1
                             np.array([0, 0, 50, 2]),      # L1 -> L0
                             np.array([0.9, 0, 100, 7]),   # L1 -> L2
                             np.array([0, 0, 400, 30])])    # L2 -> L1 

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

    def __init__(self, env_config={"render_mode":None, "size":4}):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.current_time = 0  # Start at time 0
        self.timestep = 60 
        self.num_service = env_config["size"]  # The number of services
        self.num_ctn_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (RAM, CPU, GPU)
        self.max_container = 20
        self.num_container = np.floor(self.max_container / self.num_service * np.ones(self.num_service),dtype=float) # số lượng container mỗi service
        
        self.timeout = 30  # Set timeout value = 10s
        self.container_lifetime = 3600  # Set lifetime of a container = 1 day
        self.limited_resource = [1000,1000]  # Set limited amount of [RAM,CPU] of system
        self.average_requests = 5/60  # Set the average incoming requests per second 
        self.limited_requests = int(self.average_requests*self.timestep*2)  # Set the limit number of requests that can exist in the system 
        self.energy_cost = 0.00000463333 # unit cent/Jun 
        self.ram_profit = 0.00002632022 # unit cent/Gb
        self.cpu_profit = 0.00002632022 # unit cent/vcpu
        self.num_rq_state = 4
        self.num_trans = Transitions.shape[0]
        '''
        Initialize the state and other variables
        '''
        self.truncated = False
        self.terminated = False
        # self.truncated_reason = ""
        self.temp_reward = 0 # Reward cache for each step
        self.current_resource_usage = np.zeros(self.num_resources)
        
        
        self._all_requests =  [[] for _ in range(self.num_service)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.num_service)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.num_service)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.num_service)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.num_service)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.num_service)] # Timeout requests cache in a timestep
        
        self._action_matrix = None  # Set an initial value
        
        # Cache of container and state
        # TODO: thử random các trạng thái khởi tạo ma trận container 
        # Tạo ma trận dựa trên self.num_container
        # self._container_matrix_tmp = np.hstack((
        #     self.num_container[:, np.newaxis],  # Chuyển đổi mảng thành ma trận cột
        #     np.zeros((self.num_container.size, self.num_ctn_states-1), dtype=np.int16)  # Ma trận các số 0 kích thước 4x3
        # )).astype(np.int16)
        self._container_matrix_tmp = self._create_container_matrix()
        self._container_matrix = self._container_matrix_tmp

        
        # Observation matrices cache
        self._env_matrix = np.zeros((self.num_service, self.num_ctn_states+4),dtype=np.int16)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix

        '''
        Define observations (state space)
        '''
        self.raw_state_space = self._state_space_init() 
        self.state_space = spaces.flatten_space(self.raw_state_space)
        self.state_size = self._num_state_cal()
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self.raw_action_space = self._action_space_init() 
        self.action_size = self._num_action_cal()
        self.action_space = spaces.Discrete(self.action_size,seed=42)
        self.action_mask = np.zeros((self.action_size),dtype=np.int8)
        self._cal_action_mask()
        self.formatted_action = np.zeros((2,4),dtype=np.int32)

        assert env_config["render_mode"] is None or env_config["render_mode"] in self.metadata["render_modes"]
        self.render_mode = env_config["render_mode"]

    # Tạo không gian action
    def _action_space_init(self):
        high_matrix = np.zeros((2,self.num_service),dtype=np.int16)
        for service in range(self.num_service):
            high_matrix[0][service]=self.num_container[service]
            high_matrix[1][service]=Transitions.shape[0] 
            
        action_space = spaces.Box(low=0,high=high_matrix,shape=(2,self.num_service), dtype=np.int16) # Num contaier * num transition * num service
        return action_space
    # Tính số lượng phần tử của action space
    def _num_action_cal(self):
        num_action = 1
        for service  in range(self.num_service): 
            num_action *= (1 + Transitions.shape[0]*self.num_container[service])
        return int(num_action)

    # Tạo không gian state
    def _state_space_init(self):
        low_matrix = np.zeros((self.num_service, self.num_ctn_states+4),dtype=np.int16)
        high_matrix = np.zeros((self.num_service, self.num_ctn_states+4),dtype=np.int16)
        for service in range(self.num_service):
            for container_state in range(self.num_ctn_states):
                # low_matrix[service][container_state] = -self.num_container[service]
                high_matrix[service][container_state] = 2*self.num_container[service]
            
            high_matrix[service][Request_States.Done+self.num_ctn_states] = self.limited_requests 
            high_matrix[service][Request_States.In_Queue+self.num_ctn_states] = self.limited_requests 
            high_matrix[service][Request_States.In_System+self.num_ctn_states] = self.limited_requests 
            high_matrix[service][Request_States.Time_Out+self.num_ctn_states] = self.limited_requests 
            
        state_space = spaces.Box(low=low_matrix, high=high_matrix, shape=(self.num_service, self.num_ctn_states+4), dtype=np.int16)  # num_service *(num_container_state + num_request_state)
        return state_space
    # Tính số lượng phần tử của action space
    def _num_state_cal(self):
        ret = 1
        for service in range(self.num_service):
            ret *= compute_formula(self.num_ctn_states,int(self.num_container[service])) 
        ret *= compute_formula(self.num_rq_state,int(2*self.limited_requests))
        return ret

    def _cal_action_mask(self):
        self.action_mask.fill(0)
        tmp_action_mask = np.empty((self.num_service),dtype=object) 
        for service in range(self.num_service):
            # Luôn có acton không làm gì cả
            coefficient = []
            for state in range(self.num_ctn_states-1):
                if 0 == state:
                    coefficient.append({1:self._container_matrix[service][state]})
                elif 1 == state:
                    coefficient.append({2:self._container_matrix[service][state]}) 
                    coefficient.append({3:self._container_matrix[service][state]})
                elif 2 == state:
                    coefficient.append({4:self._container_matrix[service][state]})
                    coefficient.append({5:self._container_matrix[service][state]})
                elif 3 == state:
                    coefficient.append({6:self._container_matrix[service][state]})
            # coefficient.append({6:np.int32(self.num_container[service])})
            tmp_action_mask[service] = np.array(coefficient)
        trans_combs = list(itertools.product(range(1,self.num_trans), repeat=self.num_service))       
        for trans_comb in trans_combs:
            ctn_ranges = []
            for service in range(self.num_service):
                h = tmp_action_mask[service][trans_comb[service]-1][trans_comb[service]]
                ctn_ranges.append(range(h + 1))
            for ctn_comb in itertools.product(*ctn_ranges):
                index = self.action_to_index(np.array([list(ctn_comb), list(trans_comb)]))
                self.action_mask[index] = 1
                 
    def _create_container_matrix(self):
        ret = np.zeros(shape=(len(self.num_container), self.num_ctn_states),dtype=np.int64)     
        for service in range(self.num_service):
            tmp = self.num_container[service]
            for state in range(self.num_ctn_states-2):
                if tmp > 0 :
                    ret[service][state] = np.random.randint(0,tmp)
                    tmp -= ret[service][state]
            ret[service][self.num_ctn_states-2] = tmp
        return ret
    
    # def random_maksed_action(self):
    #     masked_action = np.zeros((2,self.num_service),dtype=np.int64)
    #     for service in range(self.num_service):
    #         rand_idx = np.random.randint(len(self.action_mask[1][service]))
    #         masked_action[1][service] = self.action_mask[1][service][rand_idx]
    #         masked_action[0][service] = np.random.randint(self.action_mask[0][service][rand_idx])
    #     return masked_action
    
    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        '''    
        return spaces.flatten(self.raw_state_space,self._env_matrix)

    def _get_info(self):
        '''
        Defines a function that returns system evaluation parameters
        '''
        
        return {
            "current_step": int(self.current_time/self.timestep)
        }

    def _get_reward(self):
        # if self.truncated == True:
        #     return -10
        # else:
        return self.temp_reward
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.current_resource_usage.fill(0)

        # Reset giá trị của self._container_matrix
        self._container_matrix = self._container_matrix_tmp
        
        # Observation matrices cache
        self._env_matrix.fill(0)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        
        
        self._all_requests =  [[] for _ in range(self.num_service)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.num_service)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.num_service)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.num_service)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.num_service)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.num_service)] # Timeout requests cache in a timestep
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
     
    def _cal_system_resource(self):
        # Tính tài nguyên tiêu thụ trên toàn hệ thống sau khi thực hiện action
        current_resource_usage = [0]*self.num_resources
        for resource in range(self.num_resources):
            for service in range(self.num_service):
                # Tài nguyên tiêu thụ tức thời bởi các request trong hệ thống
                for request in self._in_system_requests[service]:
                    current_resource_usage[resource] += request.resource_usage[resource]
                
                
                for state in range(self.num_ctn_states):
                    # Tài nguyên tiêu thụ tức thời tại các trạng thái của container
                    current_resource_usage[resource] +=  self._container_matrix[service][state]*Container_Resource_Usage[state][resource] 
                    
                    # Tài nguyên tiêu thụ tức thời do chuyển trạng thái
                    trans_num  = self.formatted_action[0][service]
                    trans_type = self.formatted_action[1][service]
                    # Kiểm tra xem chuyển trạng thái xong chưa
                    if self.current_time % self.timestep < Transitions_cost[trans_type][Resource_Type.Time]:
                        current_resource_usage[resource] += Transitions_cost[trans_type][resource]*trans_num
                
            self.current_resource_usage[resource] = current_resource_usage[resource] 
                    
    def _receive_new_requests(self):
        new_requets = rq.generate_requests(size=self.num_service,current_time=self.current_time, duration=self.timestep,avg_requests_per_second=self.average_requests,timeout=self.timeout)
        for request in new_requets:
            self._new_requests[request.type].append(request)
            self._all_requests[request.type].append(request)
            self._in_queue_requests[request.type].append(request)
        # Sắp xếp lại request trong queue theo thời gian đến hệ thống => FCFS
        for service in range(self.num_service):
            self._in_queue_requests[service].sort(key=lambda x: x.in_queue_time)
            
    def _set_truncated(self):
        temp = self._container_matrix + self._action_matrix
        if (np.any(temp < 0)):
            self.truncated = True
            print(self._container_matrix)
            print(self._action_matrix)
            print(temp)
            self.truncated_reason = "Wrong number action"
        
        # Truncated nêú action làm tràn tài nguyên hệ thống
        # TODO: Phần này cần xem xét lại vì request đến cũng có thể là tràn tài nguyên hệ thống
        # self._cal_system_resource()      
        # if self.current_resource_usage[Resource_Type.RAM] > self.limited_resource[Resource_Type.RAM]:
        #     self.truncated = True
        #     self.truncated_reason = "RAM overloaded"
        # if self.current_resource_usage[ Resource_Type.CPU] > self.limited_resource[Resource_Type.CPU]:
        #     self.truncated = True
        #     self.truncated_reason = "CPU overloaded"
            
    def _set_terminated(self):
        if (self.current_time >= self.container_lifetime):
            self.terminated = True            
    
    def _apply_action(self):
        self._container_matrix += self._action_matrix     
        # print("After apply action state")
        # print(self._container_matrix )         
            
    def _handle_env_change(self):
        # Xử lý request mỗi giây 1 lần cho 1 timestep
        relative_time = 0
        while relative_time < self.timestep:
            for service in range(self.num_service):
                # Giảm time out       
                for request in self._in_queue_requests[service]:
                    if request.in_queue_time <= self.current_time:
                        # Giải phóng các request bị time_out
                        if request.time_out == 0:
                            request.set_state(Request_States.Time_Out)
                            request.set_out_system_time(self.current_time)
                            self._timeout_requests[service].append(request)
                            self._in_queue_requests[service].remove(request)
                        elif request.in_system_time <= self.current_time:
                            request.time_out -= 1
                            # Nếu còn tài nguyên trống, đẩy request vào 
                            if self._container_matrix[service][Container_States.Warm_CPU] > 0:
                                request.set_state(Request_States.In_System)
                                request.set_in_system_time(self.current_time)
                                self._in_system_requests[service].append(request)
                                self._in_queue_requests[service].remove(request)
                                self._container_matrix[service][Container_States.Active] += 1
                                self._container_matrix[service][Container_States.Warm_CPU]    -= 1
                                
                # Giảm active_time
                for request in self._in_system_requests[service]:
                    # Giải phóng các request đã thực hiện xong
                    if request.active_time == 0:
                        # print("Current time {}............".format(self.current_time))
                        # print("Type: {}".format(request.type))
                        # print("Timeout: {}".format(request.time_out))
                        # print("Active time: {}".format(request.active_time))
                        # print("Resource usage: {}".format(request.resource_usage))
                        # print("In queue time: {}".format(request.in_queue_time))
                        # print("In system time: {}".format(request.in_system_time))
                        # print("Out system time: {}".format(request.out_system_time))
                        # print("State: {}".format(request.state))
                        request.set_state(Request_States.Done)
                        request.set_out_system_time(self.current_time)
                        self._done_requests[service].append(request)
                        self._in_system_requests[service].remove(request)
                        self._container_matrix[service][Container_States.Active] -= 1
                        self._container_matrix[service][Container_States.Warm_CPU] += 1

                    else:
                        request.active_time -= 1
                
                # Tính năng lượng tiêu tốn trên toàn bộ hệ thống
                self._cal_system_resource()
                # Tính reward tức thời cho thời điểm hiện tại
                self._cal_temp_reward() 
                
                self.current_time += 1
                relative_time += 1
                
        # Tính ma trận môi trường
        self._cal_env_matrix()
    
    def _cal_env_matrix(self):
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        for service in range(self.num_service):
            for request in self._in_queue_requests[service]:
                if  0 <= self.current_time - request.in_queue_time <= self.timestep:
                    self._env_matrix[service][self.num_ctn_states+Request_States.In_Queue] += 1    
            for request in self._timeout_requests[service]:
                if 0 <= self.current_time - request.out_system_time <= self.timestep:
                    self._env_matrix[service][self.num_ctn_states+Request_States.Time_Out] += 1   
            for request in self._in_system_requests[service]:
                if 0 <= self.current_time - request.in_system_time <= self.timestep:
                    self._env_matrix[service][self.num_ctn_states+Request_States.In_System] += 1   
            for request in self._done_requests[service]:
                if 0 <= self.current_time - request.out_system_time <= self.timestep:
                    self._env_matrix[service][self.num_ctn_states+Request_States.Done] += 1 
                             
    def _cal_temp_reward(self):
        # TODO: đơn giản hóa reward, 
        # TODO: thêm based resource cho compute node chạy container để tránh tình trạng bật tất cả container warm cpu
        # TODO: nghĩ thêm về cách tính reward
        # TODO: phân tích các input có thể ảnh hướng đến kết quả
        abandone_penalty = 0
        delay_penalty = 0
        profit = 0
        energy_cost = self.current_resource_usage[Resource_Type.Power]*self.energy_cost
        
        for service in range(self.num_service):
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
    
    def _action_to_matrix(self,index):
        action = self.index_to_action(index)
        # print("hiiiiii")
        # print(action)
        action = action.reshape(2,self.num_service)
        action_coefficient = np.diag(action[0])
        action_unit = []
        for service in action[1]:
            action_unit.append(Transitions[service])
        self._action_matrix = action_coefficient @ action_unit
        return action
        
    def _clear_cache(self):
        self._new_requests = [[] for _ in range(self.num_service)] 
        
        # Observation matrix cache
        self._env_matrix.fill(0)
        # self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        
        self.action_mask.fill(0)

        self.temp_reward = 0
        self.truncated = False
        self.terminated = False
                 
       
    def _pre_step(self,action):
        self._clear_cache()
        self.formatted_action = self._action_to_matrix(action)
        self._set_terminated()
        self._set_truncated()
        if not self.truncated and not self.terminated:
        # if not self.terminated:
            self._apply_action()     # Thay đổi ma trận trạng thái của container
        
        
    def step(self, action):
        self._pre_step(action)
        self._receive_new_requests()   # Nhận request đến hệ thống trong timestep
        self._handle_env_change()     
        observation = self._get_obs()
        info = self._get_info() 
        reward = self._get_reward()
        self._cal_action_mask()
        
        return observation, reward, self.terminated, self.truncated, info
    
    def render(self):
        '''
        Implement a visualization method
        '''
        obs = self._get_obs()
        reward = self._get_reward()
        print("SYSTEM EVALUATION PARAMETERS:")
        print("- Rewards: {:.2f}".format(reward))
        print("- Energy Consumption : {:.2f}J".format(self.current_resource_usage[Resource_Type.Power]))
        print("- RAM Consumption  : {:.2f}Gb".format(self.current_resource_usage[Resource_Type.RAM]))
        print("- CPU Consumption  : {:.2f}Core".format(self.current_resource_usage[Resource_Type.CPU]))

    def close(self):
        '''
        Implement the close function to clean up (if needed)
        '''
        pass
    
    def action_to_index(self, action_matrix):
        index = 0
        multiplier = 1
        for service in range(self.num_service):
            if action_matrix[0][service] == 0:
                index += 0
            else:
                index += multiplier*(action_matrix[0][service] + (action_matrix[1][service]-1)*self.num_container[service])
            multiplier *= (self.num_container[service]*self.num_trans + 1)
        return int(index)

    def index_to_action(self, index):
        result = np.zeros((2,self.num_service),dtype=np.int32)
        tmp = 0
        multiplier = 1 
        for service in range(self.num_service-1):
            multiplier *= (self.num_container[service]*self.num_trans + 1)
            
        for service in reversed(range(self.num_service)):
            tmp = index // multiplier
            result[0][service] = tmp % self.num_container[service]
            result[1][service] = tmp // self.num_container[service]
            index %= multiplier
            multiplier //= (self.num_container[service-1]*self.num_trans + 1)
        
        return result


if __name__ == "__main__":
    # Create the serverless environment
    env = ServerlessEnv()
    # Reset the environment to the initial state
    observation, info = env.reset()
    # print(observation)
       
    # Perform random actions
    i = 0
    while (i < 1000):
        # env._cal_action_mask()
        action = env.action_space.sample(mask=env.action_mask)  # Random action
        a = env.action_mask[action]
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            print("eroor")
            break
        if (terminated): 
            print("--------------------------------cff--------")
            break
        else: continue