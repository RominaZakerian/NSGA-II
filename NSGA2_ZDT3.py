from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from deap.tools import mutPolynomialBounded
from pymoo.factory import get_crossover
from deap import tools
import numpy as np
from pymoo.interface import crossover
import pygmo as pg
import math
import random
import matplotlib.pyplot as plt

#First function to optimize
def zdt3(individual):
    """ZDT3 multiobjective function.
    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`
    :math:`f_{\\text{ZDT3}1}(\\mathbf{x}) = x_1`
    :math:`f_{\\text{ZDT3}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\sqrt{\\frac{x_1}{g(\\mathbf{x})}} - \\frac{x_1}{g(\\mathbf{x})}\\sin(10\\pi x_1)\\right]`
    """

    g  = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - math.sqrt(f1/g) - f1/g * math.sin(10*np.pi*f1))
    return f1, f2
#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossX(a,b):
    r=random.random()
    # print("nvfknf",len(a))
    # print(len(b))

    a = np.array(a)
    a=a.reshape(-1,1)
    b = np.array(b)
    b = b.reshape(-1, 1)
    # print(a)
    # print(a.shape,b.shape)
    res=crossover(get_crossover("real_sbx", prob=r,eta=20, prob_per_variable=1.0), a, b)
    # print("res",res)
    res_list=res.flatten().tolist()
    # print("res", res_list)
    # if r>0.5:
    return res_list
    # else:
    #     return mutation((a + b) / 2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    mu_solution=mutPolynomialBounded(solution, 10, min_x, max_x, mutation_prob)
    # if mutation_prob <1:
    #     solution = [min_x+(max_x-min_x)*random.random() for i in range(len(solution))]
    return mu_solution

#Main program starts here
pop_size = 100
max_gen = 1001

#Initialization
min_x=0
max_x=1
# solution=[min_x+(max_x-min_x)*random.random() for i in range(0,pop_size)]

solution_all=[]
function1_values=[]
function2_values=[]
for j in range(0,pop_size):
    solution = [min_x + (max_x - min_x) * random.random() for i in range(0, 10)]
    solution_all.append(solution)
    f1,f2=zdt3(solution)
    function1_values.append(f1)
    function2_values.append(f2)
# function1_values.append(f1)
# function2_values.append(f2)
print(function2_values)
print(function1_values)
plt.scatter(function1_values,function2_values,c="blue")
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.title("first population of ZDT3")
plt.figure()

gen_no=0
while(gen_no<max_gen):
    # function1_values , function2_values = [zdt2(solution_all[i])for i in range(0,pop_size)]
    # function2_values = [zdt2(solution_all[i])for i in range(0,pop_size)]
    # function1_values,function2_values=[zdt2(solution[i]) for i in range(0, pop_size)]
    function1_values=[]
    function2_values=[]
    points=[]
    points2=[]
    for i in range(0,pop_size):
        f1,f2=zdt3(solution_all[i])
        function1_values.append(f1)
        function2_values.append(f2)
        points.append([f1,f2])
    print("points",points)
    # non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = points)
    print("The best front for Generation number ",gen_no, " is")
    for valuez in ndf[0]:
        solution_all[valuez]=[round(elem,3) for elem in solution_all[valuez]]
        print(solution_all[valuez],end=" ")
        print("\n")
    print("\n")
    crowding_distance_values=[]
    f=[]
    # print("ndf: ", ndf)
    for i in range(0,len(ndf)):
        # crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],ndf[i][:]))
        # print("ndf",ndf)
        non_dominated=ndf[i][:]
        # print(non_dominated)
        # f2=[]
        for i in range(len(non_dominated)):
            f.append([function1_values[i],function1_values[i]])
            # f2.append(function2_values[i][:])
        # print(f)
        crowding_distance_values.append(pg.crowding_distance(points=f))
        # print("crowding_dis: ",crowding_distance_values)
    solution2 = solution_all[:][:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        result=crossX(solution_all[a1], solution_all[b1])
        # print("result",len(result))
        solution2.append(result[:10])
    # function1_values2 , function2_values2 = [zdt2(solution2[i])for i in range(0,2*pop_size)]
    # function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    function1_values2 = []
    function2_values2 = []

    # for i in range(0,pop_size):
    #     f1,f2=zdt2(solution2[i])
    #     function1_values.append(f1)
    #     function2_values.append(f2)
    #     points.append([f1,f2])
    #

    for i in range(0,2*pop_size):
        f1_2,f2_2=zdt3(solution2[i])
        function1_values2.append(f1_2)
        function2_values2.append(f2_2)
        points2.append([f1_2,f2_2])

    print("points2", points2)
    # non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    ndf_2, dl_2, dc_2, ndr_2 = pg.fast_non_dominated_sorting(points=points2)
    # non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]

    # f_2 = []
    print("ndf_2: ", ndf_2)
    for i in range(0, len(ndf_2)):
        # crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],ndf[i][:]))
        # print("ndf",ndf)
        non_dominated = ndf_2[i][:]
        print(non_dominated)
        f_2=[]
        for i in non_dominated:
            f_2.append(points2[i])
            # f2.append(function2_values[i][:])
        print("f_2",f_2)
        # print(len(f_2))
        if len(non_dominated)==1:
            crowding_distance_values2.append(np.array([0]))
        else:
            crowding_distance_values2.append(pg.crowding_distance(points=f_2))
            print("crowding_dis: ",pg.crowding_distance(points=f_2))
    # for i in range(0,len(non_dominated_sorted_solution2)):
    #     crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    # best=pg.select_best_N_mo(points=f_2, N=80)
    # print("select best",pg.select_best_N_mo(points=f_2, N=80))
    for i in range(0,len(ndf_2)):
        # print("ndf_2",ndf_2)
        non_dominated_sorted_solution2_1 = [index_of(ndf_2[i][j],ndf_2[i] ) for j in range(0,len(ndf_2[i]))]
        print("hello",non_dominated_sorted_solution2_1)
        print("hello2",crowding_distance_values2[i][:])
        sort_indexes = np.argsort(crowding_distance_values2[i][:])
        # front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        print("sorted",sort_indexes)
        print("ndf",ndf_2)
        front = [ndf_2[i][sort_indexes[j]] for j in range(0,len(ndf_2[i]))]
        print("frint",front)
        front.reverse()
        print("front",front)
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution_all=[]
    for x in new_solution:
        print(x)
        solution_all.append(solution2[x])
        print(len(solution2[x]))
        print(solution2[x])

    # solution = [solution2[i] for i in front]
    # print(len(solution),len(solution[1]))
    # print("solutions",solution)
    gen_no = gen_no + 1

#Lets plot the final front now
function1 = [i  for i in function1_values]
function2 = [j  for j in function2_values]

F=np.empty((len(function2_values),2))
for i in range(len(function2_values)):
    F[i, 0] = function1_values[i]
    F[i, 1] = function2_values[i]

print(F)
problem = get_problem("zdt3")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(),label='pf', plot_type="line", color="black", alpha=0.7)
# fig, ax = plt.subplots()
# ax.set_xlabel('Function 1', fontsize=15)
# ax.set_ylabel('Function 2', fontsize=15)
# plt.plot(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# ref_points = np.array([function1, function2])
plot.add(F,color="blue")
# ax.plot(problem.pareto_front())

# plot.title
# plot.title('Non-dominated solutions with NSGA2 after 1000 iteration' , {"pad" : 30})
plot.show()
# plt.xlabel('Function 1', fontsize=15)
# plt.ylabel('Function 2', fontsize=15)
# plt.scatter(function1, function2,c="blue")
# plt.title("Non-dominated solutions with NSGA2 after 500 iteration ")
# plt.show()