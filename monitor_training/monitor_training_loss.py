import matplotlib.pyplot as plt

sv_training_dir = '/home/user/on_gpu/slurm_info/Speaker_Verification/' + '106244_0'

loss_aggregate = []
iter_aggregate = []
with open(sv_training_dir, 'r') as sh_file:
    lines = sh_file.readlines()
    for line in lines:
        if 'TLoss' in line:
            tloss = float(line.split('\t')[-2].replace('TLoss:', ''))
            # tloss = float(line[line.find('TLoss') + 6:-2])
            loss_aggregate.append(tloss)
            iter = int(line.split('\t')[-4].split(':')[-1])
            iter_aggregate.append(iter)

plt.figure()
plt.plot(iter_aggregate, loss_aggregate)
plt.xlabel('iterations')
plt.ylabel('average loss')
plt.show()
