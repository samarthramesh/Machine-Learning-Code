import matplotlib.pyplot as plt
import math

exec(open("core.py").read())


model1 =[BernoulliArm(0.1), BernoulliArm(0.1), BernoulliArm(0.1), BernoulliArm(0.1), BernoulliArm(0.6)]
model2 =[BernoulliArm(0.1), BernoulliArm(0.2), BernoulliArm(0.5), BernoulliArm(0.8), BernoulliArm(0.95)]

algo1 = EpsilonGreedy(0.1, [], [])
algo2 = UCB1([], [])
algo3 = MyExp3(0.05, [], [])
algo4 = Exp3(0.05, [])

algos = [algo1, algo2, algo3, algo4]

#----------------------------------------------------
#Problem 1 Part 1

num_sims = 1
horizon = 1000
results1 = [test_algorithm(algo, model1, num_sims, horizon) for algo in algos]
results2 = [test_algorithm(algo, model2, num_sims, horizon) for algo in algos]


#----------------------------------------------
#Problem 1 Part 2

def chunks(lst, n):
    final = []
    for i in range(0, len(lst), n):
        final.append(lst[i:i + n])
    return final

p_num_sims = 1

presults1 = [test_algorithm(algo, model1, p_num_sims, horizon) for algo in algos]
presults2 = [test_algorithm(algo, model1, p_num_sims, horizon) for algo in algos]

allpcum1 = [result[4] for result in presults1]
allpcum2 = [result[4] for result in presults2]

avgrewcum1 = []
avgrewcum2 = []
for i in range(len(allpcum1)):
    seperated = chunks(allpcum1[i], 1000)
    sumpcum = [sum(x) for x in zip(*seperated)]
    avgpcum = [x/p_num_sims for x in sumpcum]
    avgrewcum1.append(avgpcum)

for i in range(len(allpcum2)):
    seperated = chunks(allpcum1[i], 1000)
    sumpcum = [sum(x) for x in zip(*seperated)]
    avgpcum = [x/p_num_sims for x in sumpcum]
    avgrewcum2.append(avgpcum)
    
ideal1 = [0.6*(i+1) for i in range(1000)]
ideal2 = [0.95*(i+1) for i in range(1000)]

regrets1 = []
regrets2 = []
for i in range(len(avgrewcum1)):
    regrets1.append([x-y for x,y in zip(ideal1, avgrewcum1[i])])

for i in range(len(avgrewcum2)):
    regrets2.append([x-y for x,y in zip(ideal2, avgrewcum2[i])])
    
#-------------------------------------------------------------
f_num_sims = 10
fresults1 = [test_algorithm(algo, model1, f_num_sims, horizon) for algo in algos]
fresults2 = [test_algorithm(algo, model2, f_num_sims, horizon) for algo in algos]

arm_choices_eps_1 = fresults1[0][2]
arm_choices_ucb_1 = fresults1[0][2]
arm_choices_exp_1 = fresults1[0][2]
arm_choices_gexp_1 = fresults1[0][2]

arm_choices_eps_2 = fresults2[0][2]
arm_choices_ucb_2 = fresults2[0][2]
arm_choices_exp_2 = fresults2[0][2]
arm_choices_gexp_2 = fresults2[0][2]

arm_choices_eps_1 = chunks(arm_choices_eps_1, 1000)
arm_choices_ucb_1 = chunks(arm_choices_ucb_1, 1000)
arm_choices_exp_1 = chunks(arm_choices_exp_1, 1000)
arm_choices_gexp_1 = chunks(arm_choices_gexp_1, 1000)

arm_choices_eps_2 = chunks(arm_choices_eps_2, 1000)
arm_choices_ucb_2 = chunks(arm_choices_ucb_2, 1000)
arm_choices_exp_2 = chunks(arm_choices_exp_2, 1000)
arm_choices_gexp_2 = chunks(arm_choices_gexp_2, 1000)

eps_arms_1 = [[0 for i in range(1000)] for i in range(5)]
ucb_arms_1 = [[0 for i in range(1000)] for i in range(5)]
exp_arms_1 = [[0 for i in range(1000)] for i in range(5)]
gexp_arms_1 = [[0 for i in range(1000)] for i in range(5)]

eps_arms_2 = [[0 for i in range(1000)] for i in range(5)]
ucb_arms_2 = [[0 for i in range(1000)] for i in range(5)]
exp_arms_2 = [[0 for i in range(1000)] for i in range(5)]
gexp_arms_2 = [[0 for i in range(1000)] for i in range(5)]


for sim in arm_choices_eps_1:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            eps_arms_1[arm][i] += counters[arm]
        
eps_fin_1 = [[x/f_num_sims for x in arm_choice] for arm_choice in eps_arms_1]

for sim in arm_choices_ucb_1:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            ucb_arms_1[arm][i] += counters[arm]
        
ucb_fin_1 = [[x/f_num_sims for x in arm_choice] for arm_choice in ucb_arms_1]

for sim in arm_choices_exp_1:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            exp_arms_1[arm][i] += counters[arm]
        
exp_fin_1 = [[x/f_num_sims for x in arm_choice] for arm_choice in exp_arms_1]

for sim in arm_choices_gexp_1:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            gexp_arms_1[arm][i] += counters[arm]
        
gexp_fin_1 = [[x/f_num_sims for x in arm_choice] for arm_choice in gexp_arms_1]

for sim in arm_choices_eps_2:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            eps_arms_2[arm][i] += counters[arm]
        
eps_fin_2 = [[x/f_num_sims for x in arm_choice] for arm_choice in eps_arms_2]

for sim in arm_choices_ucb_2:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            ucb_arms_2[arm][i] += counters[arm]
        
ucb_fin_2 = [[x/f_num_sims for x in arm_choice] for arm_choice in ucb_arms_2]

for sim in arm_choices_exp_2:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            exp_arms_2[arm][i] += counters[arm]
        
exp_fin_2 = [[x/f_num_sims for x in arm_choice] for arm_choice in exp_arms_2]

for sim in arm_choices_gexp_2:
    counters = [0,0,0,0,0]
    for i in range(len(sim)):
        counters[sim[i]] +=1
        for arm in range(5):
            gexp_arms_2[arm][i] += counters[arm]
        
gexp_fin_2 = [[x/f_num_sims for x in arm_choice] for arm_choice in gexp_arms_2]

#-------------------------------------------------------------
#Problem 3 Part a
delta = 0.25
best = 0.875
model3 = [BernoulliArm(0.5), BernoulliArm(0.5+delta)]

def lrfromhorizon(horizon):
    return math.sqrt(math.log(2)/horizon)

algo5 = UCB1([],[])

ucblist = []
explist = []
for i in range(100):
    ucbrew =[]
    exprew =[]
    for horizon in range(100, 1000):
        l_rate = lrfromhorizon(horizon)
        algo = MyExp3(l_rate, [], [])
        result1 = test_algorithm(algo5, model3, 1, horizon)
        result2 = test_algorithm(algo, model3, 1, horizon)
        ucbrew.append(result1[4][-1])
        exprew.append(result2[4][-1])
    ucblist.append(ucbrew)
    explist.append(exprew)

    
fin_ucb_list = [sum(x)/100 for x in zip(*ucblist)]
fin_exp_list = [sum(x)/100 for x in zip(*explist)]

ideal_rew = [best*horizon for horizon in range(100, 1000)]

ucb_regrets = [(x-y for x,y in zip(ideal_rew, fin_ucb_list))]
exp_regrets = [(x-y for x,y in zip(ideal_rew, fin_exp_list))]


#-------------------------------------------------------------
#Problem 3 Part b and c

l_rates = [(0.02*(i+1)) for i in range(50)]
best_reward = best*1000
regret_list = []
for i in range(500):
    between_list =[]
    for rate in l_rates:
        algo = MyExp3(rate, [], [])
        result = test_algorithm(algo, model3, 1, 1000)
        regret = best_reward - result[4][-1]
        between_list.append(regret)
    regret_list.append(between_list)
    print(i)
    
regret_list = [sum(x)/100 for x in zip(*regret_list)]

#-------------------------------------------------------------
#Plotting Graphs
plt.figure(1)
plt.plot(results1[0][4], color='b', label="Epsilon-Greedy")
plt.plot(results1[1][4], color='r', label="UCB")
plt.plot(results1[2][4], color='k', label="EXP3")
plt.plot(results1[3][4], color='y', label="Gamma-EXP3")
plt.legend(loc='best', shadow=True)
plt.title(label = "Model 1")

plt.figure(2)
plt.plot(results2[0][4], color='b', label="Epsilon-Greedy")
plt.plot(results2[1][4], color='r', label="UCB")
plt.plot(results2[2][4], color='k', label="EXP3")
plt.plot(results2[3][4], color='y', label="Gamma-EXP3")
plt.legend(loc='best', shadow=True)
plt.title(label = "Model 2")

plt.figure(3)
plt.plot(regrets1[0], color='b', label="Epsilon-Greedy")
plt.plot(regrets1[1], color='r', label="UCB")
plt.plot(regrets1[2], color='k', label="EXP3")
plt.plot(regrets1[3], color='y', label="Gamma-EXP3")
plt.legend(loc='best', shadow=True)
plt.title(label = "Model 1 Pseudo-Regrets")

plt.figure(4)
plt.plot(regrets2[0], color='b', label="Epsilon-Greedy")
plt.plot(regrets2[1], color='r', label="UCB")
plt.plot(regrets2[2], color='k', label="EXP3")
plt.plot(regrets2[3], color='y', label="Gamma-EXP3")
plt.legend(loc='best', shadow=True)
plt.title(label = "Model 2 Pseudo-Regrets")

plt.figure(5)
plt.plot(eps_fin_1[0], color='b', label="Arm 1")
plt.plot(eps_fin_1[1], color='r', label="Arm 2")
plt.plot(eps_fin_1[2], color='k', label="Arm 3")
plt.plot(eps_fin_1[3], color='y', label="Arm 4")
plt.plot(eps_fin_1[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for Epsilon-Greedy on Model 1")

plt.figure(6)
plt.plot(ucb_fin_1[0], color='b', label="Arm 1")
plt.plot(ucb_fin_1[1], color='r', label="Arm 2")
plt.plot(ucb_fin_1[2], color='k', label="Arm 3")
plt.plot(ucb_fin_1[3], color='y', label="Arm 4")
plt.plot(ucb_fin_1[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for UCB on Model 1")

plt.figure(7)
plt.plot(exp_fin_1[0], color='b', label="Arm 1")
plt.plot(exp_fin_1[1], color='r', label="Arm 2")
plt.plot(exp_fin_1[2], color='k', label="Arm 3")
plt.plot(exp_fin_1[3], color='y', label="Arm 4")
plt.plot(exp_fin_1[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for EXP3 on Model 1")

plt.figure(8)
plt.plot(gexp_fin_1[0], color='b', label="Arm 1")
plt.plot(gexp_fin_1[1], color='r', label="Arm 2")
plt.plot(gexp_fin_1[2], color='k', label="Arm 3")
plt.plot(gexp_fin_1[3], color='y', label="Arm 4")
plt.plot(gexp_fin_1[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for Gamma-EXP3 on Model 1")

plt.figure(9)
plt.plot(eps_fin_2[0], color='b', label="Arm 1")
plt.plot(eps_fin_2[1], color='r', label="Arm 2")
plt.plot(eps_fin_2[2], color='k', label="Arm 3")
plt.plot(eps_fin_2[3], color='y', label="Arm 4")
plt.plot(eps_fin_2[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for Epsilon-Greedy on Model 2")

plt.figure(10)
plt.plot(ucb_fin_2[0], color='b', label="Arm 1")
plt.plot(ucb_fin_2[1], color='r', label="Arm 2")
plt.plot(ucb_fin_2[2], color='k', label="Arm 3")
plt.plot(ucb_fin_2[3], color='y', label="Arm 4")
plt.plot(ucb_fin_2[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for UCB on Model 2")

plt.figure(11)
plt.plot(exp_fin_2[0], color='b', label="Arm 1")
plt.plot(exp_fin_2[1], color='r', label="Arm 2")
plt.plot(exp_fin_2[2], color='k', label="Arm 3")
plt.plot(exp_fin_2[3], color='y', label="Arm 4")
plt.plot(exp_fin_2[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for EXP3 on Model 2")

plt.figure(12)
plt.plot(gexp_fin_2[0], color='b', label="Arm 1")
plt.plot(gexp_fin_2[1], color='r', label="Arm 2")
plt.plot(gexp_fin_2[2], color='k', label="Arm 3")
plt.plot(gexp_fin_2[3], color='y', label="Arm 4")
plt.plot(gexp_fin_2[4], color='g', label="Arm 5")
plt.legend(loc='best', shadow=True)
plt.title(label = "Arm Choices for Gamma-EXP3 on Model 2")

plt.figure(13)
plt.plot(fin_ucb_list, list(range(100, 1000)), color = 'b', label="UCB")
plt.plot(fin_exp_list, list(range(100, 1000)), color = 'r', label="EXP3")
plt.legend(loc='best', shadow=True)
plt.title(label = "Regret vs Horizon")

plt.figure(14)
plt.plot(l_rates, regret_list, color = 'b')
plt.title(label="Regret vs Learning Rate for EXP3(Delta=0.2)")

plt.show()