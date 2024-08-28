import re
import os
import matplotlib.pyplot as plt
import math

# Đường dẫn đến thư mục chứa file log
log_folder = '/home/mec/hai/RL-for-serverless/envs/result/result_36_14_19_27_8/test'
log_file = os.path.join(log_folder, 'log.txt')

# Đọc file log
with open(log_file, 'r') as file:
    log_data = file.read()
    

training_num_pattern = re.compile(r'Test trainned model (\d+) times') 
training_num_match = training_num_pattern.search(log_data)
eps = int(training_num_match.group(1))
# Lấy giá trị timestep từ ENVIRONMENT PARAMETERS
timestep_pattern = re.compile(r'"timestep": (\d+),')
timestep_match = timestep_pattern.search(log_data)
if timestep_match:
    timestep_value = int(timestep_match.group(1))
else:
    timestep_value = 120  # Giá trị mặc định nếu không tìm thấy

# Tìm các khối "SYSTEM EVALUATION PARAMETERS IN TIMESTEP"
timestep_blocks = re.findall(r'SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:.*?(?=SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:|\Z)', log_data, re.S)

# Khởi tạo các danh sách để lưu trữ giá trị
timesteps = []
new_rqs = []
in_queue_rqs = []
in_sys_rqs = []
done_rqs = []
rewards = []
energy_consumptions = []

# Trích xuất các thông số từ mỗi khối
for i, block in enumerate(timestep_blocks, start=1):
    new_rq_match = re.search(r'Number new request\s*:\s*(\d+)', block)
    new_rq = int(new_rq_match.group(1)) if new_rq_match else None
    
    in_queue_rq_match = re.search(r'Number in queue request\s*:\s*(\d+)', block)
    in_queue_rq = int(in_queue_rq_match.group(1)) if in_queue_rq_match else None
    
    in_sys_rq_match = re.search(r'Number in system request\s*:\s*(\d+)', block)
    in_sys_rq = int(in_sys_rq_match.group(1)) if in_sys_rq_match else None
    
    # Tìm Cumulative number accepted request
    done_rq_match = re.search(r'Number done system request\s*:\s*(\d+)', block)
    done_rq = int(done_rq_match.group(1)) if done_rq_match else None
    
    # Tìm Rewards
    rewards_match = re.search(r'Rewards\s*:\s*([-\d.]+)', block)
    reward = float(rewards_match.group(1)) if rewards_match else None
    
    # Tìm Energy Consumption
    energy_match = re.search(r'Energy Consumption\s*:\s*([\d.]+)J', block)
    energy_consumption = float(energy_match.group(1)) if energy_match else None
    
    # Lưu các thông số vào danh sách
    timesteps.append(i * timestep_value)  # Tính toán thời gian (giây)
    new_rqs.append(new_rq)
    in_queue_rqs.append(in_queue_rq)
    in_sys_rqs.append(in_sys_rq)
    done_rqs.append(done_rq)
    rewards.append(reward)
    energy_consumptions.append(energy_consumption)

# Tính toán accepted ratio
acceptance_ratio = [
    (in_sys_rqs[i] + done_rqs[i] - (in_sys_rqs[i-1] if i > 0 else 0)) /
    ((in_queue_rqs[i-1] if i > 0 else 0) + new_rqs[i])
    for i in range(len(new_rqs))
]

print(len(acceptance_ratio))

# Vẽ accepted ratio theo thời gian
plt.figure()
plt.plot(timesteps, acceptance_ratio, label='Acceptance Ratio', color='purple')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceptance Ratio')
plt.title(' Acceptance Ratio Over Time')
plt.grid(True)
plt.savefig(os.path.join(log_folder, 'acceptance_ratio_plot.png'))  # Lưu hình
plt.show()


# Vẽ hình các metrics khác (tương tự như trước)
# Hình Rewards
plt.figure()
plt.plot(timesteps, rewards, label='Rewards', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Rewards')
plt.title('Rewards Over Time')
plt.grid(True)
plt.savefig(os.path.join(log_folder, 'rewards_plot.png'))  # Lưu hình
plt.show()

# Hình Energy Consumption
plt.figure()
plt.plot(timesteps, energy_consumptions, label='Energy Consumption', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy Consumption (J)')
plt.title('Energy Consumption Over Time')
plt.grid(True)
plt.savefig(os.path.join(log_folder, 'energy_consumption_plot.png'))  # Lưu hình
plt.show()
