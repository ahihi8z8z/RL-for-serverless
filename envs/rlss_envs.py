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
 Request type: [RAM, CPU, Power]  
'''      
Request_Resource_Usage = np.array([np.array([10, 10, 100]),   
                          np.array([20, 20, 200]),   
                          np.array([30, 30, 300]),   
                          np.array([40, 40, 400])])   
    
'''
Define cases where state changes can occur:
    N <-> L0 <-> L1 <-> L2 <-> A
'''    
Transitions = np.array([np.array([0, 0, 0, 0, 0]),    # No change
                        np.array([-1, 1, 0, 0, 0]),   # N -> L0
                        np.array([-1, 0, 0, 1, 0]),   # N -> L2 (nhảy cóc)
                        np.array([1, -1, 0, 0, 0]),   # L0 -> N
                        np.array([0, -1, 1, 0, 0]),   # L0 -> L1
                        np.array([0, -1, 0, 1, 0]),   # L0 -> L2 (nhảy cóc)
                        np.array([0, 1, -1, 0, 0]),   # L1 -> L0
                        np.array([0, 0, -1, 1, 0]),   # L1 -> L2
                        np.array([0, 0, 1, -1, 0]),   # L2 -> L1
                        ])

State_trans_mapping =np.array([np.array([1,2]),                 # N
                                np.array([3,4,5]),              # L0
                                np.array([6,7]),                # L1
                                np.array([8])],dtype=object)    # L2   
    
'''
Define transition cost for moving to another states:
 state: [RAM, CPU, Power, Time]  
'''    
Transitions_cost = np.array([np.array([1.2, 0, 0, 0]),         # No change 
                             np.array([0, 0, 50, 5]),          # N -> L0
                             np.array([0.9, 0, 2150, 62]),     # N -> L2
                             np.array([0, 0, 50, 2]),          # L0 -> N
                             np.array([0, 0, 2000, 50]),       # L0 -> L1
                             np.array([0.9, 0, 2100, 57]),     # L0 -> L2
                             np.array([0, 0, 50, 2]),          # L1 -> L0
                             np.array([0.9, 0, 100, 7]),       # L1 -> L2
                             np.array([0, 0, 400, 30])])       # L2 -> L1 

'''
Define resource usage for staying in each state:
 state: [RAM, CPU, Power]
    
'''    
Container_Resource_Usage =np.array([np.array([10, 10, 1000]),   # N
                          np.array([20, 20, 2000]),             # L0
                          np.array([30, 30, 3000]),            # L1
                          np.array([40, 40, 4000]),            # L2
                          np.array([40, 40, 4000])])           # A


class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, env_config={"render_mode":None, "size":1, "log_file": "log.txt"}):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.current_time = 0  # Start at time 0
        self.timestep = 120 
        
        self.num_service = env_config["size"]  # The number of services
        self.num_ctn_states = len([attr for attr in vars(Container_States) if not attr.startswith('__')])  
        self.num_trans = Transitions.shape[0] - 1
        
        self.max_container = 50
        self.num_container = np.floor(self.max_container / self.num_service * np.ones(self.num_service),dtype=float) # số lượng container mỗi service
        self.num_container.astype(int)
        self.container_lifetime = 3600*4  # Set lifetime of a container  
        
        self.num_rq_state = len([attr for attr in vars(Container_States) if not attr.startswith('__')])         
        self.rq_timeout = 10  
        self.max_rq_active_time = 120 # Set to 0 for static request active time 240s
        self.average_requests = 5/60  # Set the average incoming requests per second 
        self.max_num_request = int(self.average_requests*self.timestep*2)  # Set the limit number of requests that can exist in the system 
        
        self.num_resources = len([attr for attr in vars(Resource_Type) if not attr.startswith('__')]) - 1    # The number of resource parameters (RAM, CPU, Power)
        self.limited_resource = [1000,1000]  # Set limited amount of [RAM,CPU] of system
        self.energy_price = 10e-8 # unit cent/Jun/s 
        self.ram_profit = 100*10e-8 # unit cent/Gb/s
        self.cpu_profit = 100*10e-8 # unit cent/vcpu/s
        
        '''
        Initialize the state and other variables
        '''
        self.truncated = False
        self.terminated = False
        self.truncated_reason = ""
        self.temp_reward = 0 # Reward for each step
        self.abandone_penalty = 0
        self.delay_penalty = 0
        self.profit = 0
        self.energy_cost = 0
        self.current_resource_usage = np.zeros(self.num_resources)
        
        
        self._all_requests =  [[] for _ in range(self.num_service)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.num_service)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.num_service)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.num_service)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.num_service)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.num_service)] # Timeout requests cache in a timestep
        
        self.current_action = 0
        self._action_matrix = np.zeros(shape=(self.num_service,self.num_ctn_states)) 
        self._positive_action_matrix = self._action_matrix * (self._action_matrix > 0)
        self._negative_action_matrix = self._action_matrix * (self._action_matrix < 0)
        self.formatted_action = np.zeros((2,4),dtype=np.int32)
        
        # TODO: thử random các trạng thái khởi tạo ma trận container 
        # Tạo ma trận dựa trên self.num_container
        self._container_matrix_tmp = np.hstack((
            self.num_container[:, np.newaxis],  # Chuyển đổi mảng thành ma trận cột
            np.zeros((self.num_container.size, self.num_ctn_states-1), dtype=np.int16)  # Ma trận các số 0 kích thước 4x3
        )).astype(np.int16)
        # self._container_matrix_tmp = self._create_random_container_matrix()
        self._container_matrix = self._container_matrix_tmp.copy()


        # State space
        self.raw_state_space = self._state_space_init() 
        self.state_space = spaces.flatten_space(self.raw_state_space)
        self.state_size = self.state_space.shape[0]
        
        # State matrices cache
        self._env_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix

        # Action space
        self.raw_action_space = self._action_space_init() 
        self.action_size = self._num_action_cal()
        self.action_space = spaces.Discrete(self.action_size,seed=42)
        
        # Action masking
        self.action_mask = np.zeros((self.action_size),dtype=np.int8)
        self._cal_action_mask()

        assert env_config["render_mode"] is None or env_config["render_mode"] in self.metadata["render_modes"]
        self.render_mode = env_config["render_mode"]
        self.log_file = env_config["log_file"]
        self._print_env_params()

    def _print_env_params(self):
        with open(self.log_file, 'w') as f:
            f.write("ENVIRONMENT PARAMETERS:\n")
            f.write("- Timestep: {}\n".format(self.timestep))
            f.write("- Container life time: {}\n".format(self.container_lifetime))
            f.write("- Number service: {}\n".format(self.num_service))
            f.write("- Number container each service: {}\n".format(self.num_container))
            f.write("- Number request per timestep: {}\n".format(self.average_requests*self.timestep))
            f.write("\n")
            
            
    # Tạo không gian action
    def _action_space_init(self):
        high_matrix = np.zeros((2,self.num_service),dtype=np.int16)
        for service in range(self.num_service):
            high_matrix[0][service]=self.num_container[service]
            high_matrix[1][service]= self.num_trans 
            
        action_space = spaces.Box(low=1,high=high_matrix,shape=(2,self.num_service), dtype=np.int16) # Num contaier * num transition * num service
        return action_space
    
    # Tính số lượng phần tử của action space
    def _num_action_cal(self):
        num_action = 1
        for service  in range(self.num_service): 
            num_action *= (1 + self.num_trans*self.num_container[service])
        return int(num_action)

    # Tạo không gian state
    def _state_space_init(self):
        low_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        high_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        for service in range(self.num_service):
            for container_state in range(self.num_ctn_states):
                # low_matrix[service][container_state] = -self.num_container[service]
                high_matrix[service][container_state] = 2*self.num_container[service]
            
            # high_matrix[service][Request_States.Done+self.num_ctn_states] = self.max_num_request 
            high_matrix[service][Request_States.In_Queue+self.num_ctn_states] = self.max_num_request 
            # high_matrix[service][Request_States.In_System+self.num_ctn_states] = self.max_num_request 
            # high_matrix[service][Request_States.Time_Out+self.num_ctn_states] = self.max_num_request 
            
        state_space = spaces.Box(low=low_matrix, high=high_matrix, shape=(self.num_service, self.num_ctn_states+1), dtype=np.int16)  # num_service *(num_container_state + num_request_state)
        return state_space
    
    # Tính số lượng phần tử của action space
    def _num_state_cal(self):
        ret = 1
        for service in range(self.num_service):
            ret *= compute_formula(self.num_ctn_states,int(self.num_container[service])) 
        ret *= compute_formula(self.num_rq_state,int(2*self.max_num_request))
        return ret

    def _cal_action_mask(self):
        self.action_mask.fill(0)
        self.action_mask[0] = 1
        tmp_action_mask = np.empty((self.num_service),dtype=object) 
        for service in range(self.num_service):
            coefficient = []
            for state in range(self.num_ctn_states-1):
                for trans in State_trans_mapping[state]:
                    coefficient.append({trans:self._container_matrix[service][state]})
            tmp_action_mask[service] = np.array(coefficient)
        
        trans_combs = list(itertools.product(range(1,self.num_trans+1), repeat=self.num_service)) 
        for trans_comb in trans_combs:
            ctn_ranges = []
            for service in range(self.num_service):
                h = tmp_action_mask[service][trans_comb[service]-1][trans_comb[service]]
                ctn_ranges.append(range(1,h + 1))
            for ctn_comb in itertools.product(*ctn_ranges):
                index = self.action_to_number(np.array([list(ctn_comb), list(trans_comb)]))
                self.action_mask[index] = 1
                 
    def _create_random_container_matrix(self):
        ret = np.zeros(shape=(len(self.num_container), self.num_ctn_states),dtype=np.int64)     
        for service in range(self.num_service):
            tmp = self.num_container[service]
            for state in range(self.num_ctn_states-2):
                if tmp > 0 :
                    ret[service][state] = np.random.randint(0,tmp)
                    tmp -= ret[service][state]
            ret[service][self.num_ctn_states-2] = tmp
        return ret
    
    
    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        ''' 
        # Tính ma trận môi trường
        self._cal_env_matrix()   
        return spaces.flatten(self.raw_state_space,self._env_matrix)

    # def _get_info(self):
    #     '''
    #     Defines a function that returns system evaluation parameters
    #     '''
        
    #     return {
    #         "current_step": int(self.current_time/self.timestep)
    #     }

    def _get_reward(self):

        self.temp_reward = self.profit - 0.1*(0.05*self.delay_penalty + 0.05*self.abandone_penalty + 0.9*self.energy_cost)
        return self.temp_reward
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.current_resource_usage.fill(0)

        # Reset giá trị của self._container_matrix
        self._container_matrix = self._container_matrix_tmp.copy()
        
        # Observation matrices cache
        self._env_matrix.fill(0)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        
        self.action_mask.fill(0)
        
        self._all_requests =  [[] for _ in range(self.num_service)] # All request come into system
        self._in_queue_requests = [[] for _ in range(self.num_service)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.num_service)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.num_service)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.num_service)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.num_service)] # Timeout requests cache in a timestep
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        
        return observation
     
    def _cal_system_resource(self, relative_time):
        # Tính tài nguyên tiêu thụ trên toàn hệ thống sau khi thực hiện action
        self.current_resource_usage.fill(0)
        for resource in range(self.num_resources):
            for service in range(self.num_service):
                # Tài nguyên tiêu thụ tức thời bởi các request trong hệ thống
                for request in self._in_system_requests[service]:
                    self.current_resource_usage[resource] += request.resource_usage[resource]
                
                for state in range(self.num_ctn_states):
                    # Tài nguyên tiêu thụ tức thời tại các trạng thái của container
                    self.current_resource_usage[resource] +=  self._container_matrix[service][state]*Container_Resource_Usage[state][resource] 
                    
                    # Tài nguyên tiêu thụ tức thời do chuyển trạng thái
                    trans_num  = self.formatted_action[0][service]
                    trans_type = self.formatted_action[1][service]
                    # Kiểm tra xem chuyển trạng thái xong chưa
                    if relative_time < Transitions_cost[trans_type][Resource_Type.Time]:
                        self.current_resource_usage[resource] += Transitions_cost[trans_type][resource]*trans_num 
                    
    def _receive_new_requests(self):
        new_requets = rq.generate_requests(size=self.num_service,
                                           current_time=self.current_time, 
                                           duration=self.timestep,
                                           avg_requests_per_second=self.average_requests,
                                           timeout=self.rq_timeout,
                                           max_rq_active_time=self.max_rq_active_time)
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
            self.truncated_reason = "Wrong number action"
            print(self.truncated_reason)
            print(self._container_matrix)
            print(self._action_matrix)
            print(self.current_action)
            print(env.action_mask[self.current_action])
            print(self.number_to_action(self.current_action))
            print(self.current_time)
        
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
            
    def _handle_env_change(self):
        # Xử lý request mỗi giây 1 lần cho 1 timestep
        self._container_matrix += self._negative_action_matrix
        relative_time = 0
        while relative_time < self.timestep:
            for service in range(self.num_service):
                # Chuyển trạng thái container 
                trans_type = self.formatted_action[1][service]
                if relative_time == Transitions_cost[trans_type][Resource_Type.Time]:
                    self._container_matrix[service] += self._positive_action_matrix[service]
                          
                for request in self._in_queue_requests[service]:
                    if request.in_queue_time <= self.current_time:
                        # Giải phóng các request bị time_out
                        if request.time_out == self.current_time - request.in_queue_time:
                            request.set_state(Request_States.Time_Out)
                            request.set_out_system_time(self.current_time)
                            self._timeout_requests[service].append(request)
                            self._in_queue_requests[service].remove(request)
                        else:
                            # Nếu còn tài nguyên trống, đẩy request vào 
                            if self._container_matrix[service][Container_States.Warm_CPU] > 0:
                                request.set_state(Request_States.In_System)
                                request.set_in_system_time(self.current_time)
                                self._in_system_requests[service].append(request)
                                self._in_queue_requests[service].remove(request)
                                self._container_matrix[service][Container_States.Active] += 1
                                self._container_matrix[service][Container_States.Warm_CPU] -= 1
                                
                for request in self._in_system_requests[service]:
                    # Giải phóng các request đã thực hiện xong
                    if request.active_time == self.current_time - request.in_system_time:
                        request.set_state(Request_States.Done)
                        request.set_out_system_time(self.current_time)
                        self._done_requests[service].append(request)
                        self._in_system_requests[service].remove(request)
                        self._container_matrix[service][Container_States.Active] -= 1
                        self._container_matrix[service][Container_States.Warm_CPU] += 1
                
                # Tính năng lượng tiêu tốn trên toàn bộ hệ thống
                self._cal_system_resource(relative_time)
                # Tính reward tức thời cho thời điểm hiện tại
                self._cal_temp_reward() 
                
            self.current_time += 1
            relative_time += 1
    
    def  _cal_env_matrix(self):
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        for service in range(self.num_service):
            for request in self._in_queue_requests[service]:
                if  0 <= self.current_time - request.in_queue_time <= self.timestep:
                    self._env_matrix[service][self.num_ctn_states+Request_States.In_Queue] += 1    
                             
    def _cal_temp_reward(self):
        # TODO: đơn giản hóa reward, 
        # TODO: thêm based resource cho compute node chạy container để tránh tình trạng bật tất cả container warm cpu
        # TODO: nghĩ thêm về cách tính reward
        # TODO: phân tích các input có thể ảnh hướng đến kết quả
        self.energy_cost += self.current_resource_usage[Resource_Type.Power]*self.energy_price
        
        for service in range(self.num_service):
            for request in self._in_system_requests[service]:
                if request.in_system_time == self.current_time:
                    # Delay penalty ở đây hiểu là tiền bị thiệt hại do request không được phục vụ ngay mà phải nằm trong queue
                    # Delay penalty chỉ tính cho nhhững request được phục vụ bởi hệ thống, những request bị timeout sẽ tính vào abandone penalty
                    # Delay penalty được một lần duy nhất tại thời điểm request được hệ thống accept
                    delay_time = request.in_system_time - request.in_queue_time
                    self.delay_penalty += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*delay_time
                    self.delay_penalty += Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit*delay_time
                else:
                    # Profit của các request được chấp nhận vào hệ thống trong 1 giây
                    self.profit += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit
                    self.profit += Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit
                
            for request in self._timeout_requests[service]:
                if request.out_system_time == self.current_time:
                    # Abandone penalty được một lần duy nhất tại thời điểm request hết timeout và bị hệ thống reject
                    in_queue_time = request.out_system_time - request.in_queue_time
                    self.abandone_penalty += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*in_queue_time
                    self.abandone_penalty += Request_Resource_Usage[service][Resource_Type.CPU]*self.ram_profit*in_queue_time
    
    def _action_to_matrix(self,index):
        self.current_action = index
        action = self.number_to_action(index)
        action = action.reshape(2,self.num_service)
        action_coefficient = np.diag(action[0])
        action_unit = []
        for service in action[1]:
            action_unit.append(Transitions[service])
        self._action_matrix = action_coefficient @ action_unit
        self._positive_action_matrix = self._action_matrix * (self._action_matrix > 0)
        self._negative_action_matrix = self._action_matrix * (self._action_matrix < 0)
        return action
        
    def _clear_cache(self):
        self._new_requests = [[] for _ in range(self.num_service)] 
        self._done_requests = [[] for _ in range(self.num_service)] 
        self._timeout_requests = [[] for _ in range(self.num_service)] 
        self._env_matrix.fill(0)
        self.action_mask.fill(0)
        self.temp_reward = 0
        self.abandone_penalty = 0
        self.delay_penalty = 0
        self.profit = 0
        self.energy_cost = 0
        self.truncated = False
        self.terminated = False
                 
    def _pre_step(self,action):
        self._clear_cache()
        self.formatted_action = self._action_to_matrix(action)
        self._set_terminated()
        self._set_truncated()
        # if not self.truncated and not self.terminated:
        # # if not self.terminated:
        #     self._apply_action()     # Thay đổi ma trận trạng thái của container
        
        
    def step(self, action):
        self._pre_step(action)
        self._receive_new_requests()   # Nhận request đến hệ thống trong timestep
        self._handle_env_change()     
        observation = self._get_obs()
        reward = self._get_reward()
        self._cal_action_mask()
        
        return observation, reward, self.terminated, self.truncated
    
    def render(self):
        '''
        Implement a visualization method
        '''
        

        with open(self.log_file, 'a') as f:
            f.write("-------------------------------------------------------------\n")
            f.write("SYSTEM EVALUATION PARAMETERS IN TIMESTEP {}: \n".format(self.current_time // self.timestep))
            f.write("- Action number: {}\n".format(self.current_action))
            f.write("- Action matrix: \n{}\n".format(self._action_matrix))
            f.write("- Containers state after action: \n{}\n".format(self._container_matrix))
            f.write("- Number new request: {}\n".format(sum(len(rqs) for rqs in self._new_requests)))
            f.write("- Number in queue request: {}\n".format(sum(len(rqs) for rqs in self._in_queue_requests)))
            f.write("- Number in system request: {}\n".format(sum(len(rqs) for rqs in self._in_system_requests)))
            f.write("- Number done system request: {}\n".format(sum(len(rqs) for rqs in self._done_requests)))
            f.write("- Number timeout system request: {}\n".format(sum(len(rqs) for rqs in self._timeout_requests)))
            f.write("- Rewards: {:.2f} = ".format(self.temp_reward))
            f.write("(Profit: {:.2f}) - ".format(self.profit))
            f.write("α*(Abandone penalty: {:.2f}) - ".format(self.abandone_penalty))
            f.write("β*(Delay penalty: {:.2f}) - ".format(self.delay_penalty))
            f.write("γ*(Energy cost: {:.2f})\n".format(self.energy_cost))
            f.write("Energy Consumption : {:.2f}J, ".format(self.current_resource_usage[Resource_Type.Power]))
            f.write("RAM Consumption  : {:.2f}Gb, ".format(self.current_resource_usage[Resource_Type.RAM]))
            f.write("CPU Consumption  : {:.2f}Core \n".format(self.current_resource_usage[Resource_Type.CPU]))
            f.write("\n")
    def action_to_number(self, action_matrix):
        index = 0
        multiplier = 1
        for service in range(self.num_service):
            if action_matrix[0][service] == 0:
                index += 0
            else:
                index += multiplier*(action_matrix[0][service] + (action_matrix[1][service]-1)*self.num_container[service])
            multiplier *= (self.num_container[service]*self.num_trans + 1)
        return int(index)

    def number_to_action(self, index):
        result = np.zeros((2,self.num_service),dtype=np.int32)
        tmp = 0
        multiplier = 1 
        for service in range(self.num_service-1):
            multiplier *= (self.num_container[service]*self.num_trans + 1)
            
        for service in reversed(range(self.num_service)):
            tmp = index // multiplier 
            if tmp == 0:
                result[0][service] = 0
                result[1][service] = 0
            else:
                result[0][service] = ((tmp-1) % self.num_container[service]) + 1
                result[1][service] = ((tmp-1) // self.num_container[service]) + 1
            index %= multiplier
            multiplier //= (self.num_container[service-1]*self.num_trans + 1)
        
        return result


if __name__ == "__main__":
    # Create the serverless environment
    env = ServerlessEnv()
    print(env._container_matrix)
    # Reset the environment to the initial state
    observation = env.reset()
    # Perform random actions
    i = 0
    while (i<10000):
        # env._cal_action_mask()
        action = env.action_space.sample(mask=env.action_mask)  # Random action
        observation, reward, terminated, truncated = env.step(action)
        env.render()
        i += 1
        if truncated:
            print("error")
            break
        if (terminated): 
            print("--------------------------------cff--------")
            env.reset()
        else: continue
    