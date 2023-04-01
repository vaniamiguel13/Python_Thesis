import math
from matplotlib import pyplot as plt
from HGPSAL.HGPSAL.auxiliares import *


def rGA(Problem, InitialPopulation=None, *varargin):
    # Define default options
    DefaultOpt = {
        'MaxObj': 2000,
        'MaxGen': 2000,
        'PopSize': 40,
        'EliteProp': 0.1,
        'TourSize': 2,
        'Pcross': 0.9,
        'Icross': 20,
        'Pmut': 0.1,
        'Imut': 20,
        'CPTolerance': 1.0e-6,
        'CPGenTest': 0.01,
        'Verbosity': False
    }
    # With no arguments just print an error

    if not Problem and not InitialPopulation and not varargin:
        raise ValueError('Invalid number of arguments. Type "help rGA" to obtain help.')

    # If just 'defaults' passed in, return the default options in X
    if Problem == 'defaults' and not InitialPopulation and not varargin:
        return DefaultOpt

    # Check parameters consistence
    if not Problem:
        raise ValueError('rGA requests at least one input (Problem definition).')

    # If parameters are missing just define them as empty
    if not varargin:
        varargin = {}
        if not InitialPopulation:
            InitialPopulation = np.array([])

    # Problem should be a dictionary
    if not isinstance(Problem, dict):
        raise ValueError('First parameter must be a dictionary.')

    # Do some sanity checkup on the user provided parameters and data
    # We need a objective function
    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('Objective function name is missing.')

    # and simple bound for genetic algorithm
    if not isinstance(Problem['LB'], (int, float, list, np.ndarray)) or \
            not isinstance(Problem['UB'], (int, float, list, np.ndarray)) or \
            not Problem['LB'] or not Problem['UB']:
        raise ValueError('Population relies on finite bounds on all variables.')

    # Bound arrays must have the same size
    if len(Problem['LB']) != len(Problem['UB']):
        raise ValueError('Lower bound and upper bound arrays length mismatch.')

    # Compute the number of variables
    # If the user provides these number check for correctness, otherwise try to
    # compute them from the size of the bound constraints
    if ('Variables' not in Problem) or (not Problem['Variables']):
        # default is to consider all variables as real
        Problem['Variables'] = len(Problem['LB'])

    # Check for computed number of variables
    if Problem['Variables'] < 0 or Problem['Variables'] > len(Problem['LB']):
        raise ValueError('Number of variables do not agree with bound constraints')

    # Start clock
    tic()
    ''' 
    Adicionar a função
    '''

    # Initialize options. GetOption returns the user specified value, if the
    # option exists. Otherwise returns the default option.

    # These are local options. Not needed for the subrotines
    MaxGenerations = GetOption('MaxGen', varargin, DefaultOpt)
    MaxEvals = GetOption('MaxObj', varargin, DefaultOpt)
    Pop = GetOption('PopSize', varargin, DefaultOpt)
    Elite = GetOption('EliteProp', varargin, DefaultOpt)
    Tour = GetOption('TourSize', varargin, DefaultOpt)
    Pc = GetOption('Pcross', varargin, DefaultOpt)
    Ic = GetOption('Icross', varargin, DefaultOpt)
    Pm = GetOption('Pmut', varargin, DefaultOpt)
    Im = GetOption('Imut', varargin, DefaultOpt)

    # These are global options. Need to be available to subrotines.
    Problem['Verbose'] = GetOption('Verbosity', varargin, DefaultOpt)
    Problem['Tolerance'] = GetOption('CPTolerance', varargin, DefaultOpt)
    Problem['GenTest'] = GetOption('CPGenTest', varargin, DefaultOpt)

    # Number of objective function calls. This is the number of calls to
    # Problem.ObjFunction.
    Problem['Stats']['ObjFunCounter'] = 0

    # Generate initial population. Include initial population provided by user
    # if available
    [Problem, Population] = InitPopulation(Problem, InitialPopulation, Pop, *varargin)

    temp = np.concatenate((Population['x'], Population['f']), axis=1)
    temp = temp[temp[:, -1].argsort()]
    Population['x'] = temp[:, :-1]
    Population['f'] = temp[:, -1]

    # if Problem['Verbose']:
    #     print('rGA is alive... ')
    #     # Search illustration: plots the population if number of variables is 2
    #     if Problem['Verbose'] == 2:
    #         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    #         ax1.set_title('Objective function')
    #         xx, yy = np.meshgrid(
    #             np.arange(Problem['LB'][0], Problem['UB'][0], (Problem['UB'][0] - Problem['LB'][0]) / 80),
    #             np.arange(Problem['LB'][1], Problem['UB'][1], (Problem['UB'][1] - Problem['LB'][1]) / 80)
    #         )
    #         z = Problem['ObjFunction'](
    #             np.vstack((xx.flatten(), yy.flatten())),
    #             *varargin
    #         ).reshape(xx.shape)
    #         ax1.plot_surface(xx, yy, z, cmap='jet')
    #         ax1.set_xlabel('x')
    #         ax1.set_ylabel('y')
    #         ax1.set_zlabel('f(x)')
    #         ax2.set_title(f'Population at generation: {0}')
    #         ax2.set_xlabel('x')
    #         ax2.set_ylabel('y')
    #         ax2.contour(xx, yy, z)
    #         p = ax2.plot(
    #             Population['x'][:, 0], Population['x'][:, 1], '.', alpha=0.5
    #         )
    #         ax2.grid(True)
    #         fig.show()

    Problem["Stats"]["GenCounter"] = 0
    Problem["Stats"]["Best"] = [Population["f"][0]]
    Problem["Stats"]["Worst"] = [Population["f"][-1]]
    Problem["Stats"]["Mean"] = [np.mean(Population["f"])]
    Problem["Stats"]["Std"] = [np.std(Population["f"])]

    while Problem["Stats"]["GenCounter"] < MaxGenerations and Problem["Stats"]["ObjFunCounter"] < MaxEvals:
        # Stop if the improvement is inferior to the Tolerance in the last generations
        if Problem["Stats"]["GenCounter"] > 0 and (
                Problem["Stats"]["GenCounter"] % math.ceil(Problem["GenTest"] * MaxGenerations) == 0) and \
                abs(Problem["Stats"]["Best"][-1] - Problem["Stats"]["Best"][
                    -1 - math.ceil(Problem["GenTest"] * MaxGenerations)]) < Problem["Tolerance"]:
            print(
                'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest '
                'generations')
            break

        # Increment generation counter.
        Problem["Stats"]["GenCounter"] += 1

        # Select the parents
        # Parents are selected for reproduction to generate offspring. The arguments are
        # pool - size of the mating pool. It is common to have this to be equal to the
        #        population size. However, if elistm is intended then pool must
        #        be inferior to the population size, i.e., pool=pop-elitesize.
        #        Typically, 10% percent of population size.
        # tour - Tournament size. For binary tournament
        #        selection set tour=2, but to see the effect of tournament size in the selection pressure this is kept
        #        arbitary, to be choosen by the user.
        elitesize = round(Pop * Elite)
        pool = Pop - elitesize
        parent_chromosome = tournament_selection(Population, pool, Tour)

        # Perform Crossover and Mutation operator
        # Simulated Binary Crossover (SBX) and
        # Polynomial mutation is used. In general, crossover probability is pc = 0.9 and mutation
        # probability is pm = 1/n, where n is the number of decision variables.
        # Typical values for the distribution indices for crossover and mutation operators are mu = 20
        # and mum = 20 respectively.
        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)

        # The offspring population replaces the worst chromosomes of the
        # population, i.e., the elite is maintained
        Population["x"][elitesize:, :] = offspring_chromosome["x"][:pool, :]

        # Evaluate the objective function
        for i in range(elitesize, Pop):
            Problem, Population["f"][i] = ObjEval(Problem, Population["x"][i, :], *varargin)

        # The new population is sorted again
        temp = np.hstack((Population["x"], Population["f"].reshape((-1, 1))))
        temp = temp[temp[:, -1].argsort()]
        Population["x"] = temp[:, :-1]
        Population["f"] = temp[:, -1]

        # Statistics
        Problem["Stats"]["Best"].append(Population["f"][0])
        Problem["Stats"]["Worst"].append(Population["f"][-1])
        Problem["Stats"]["Mean"].append(np.mean(Population["f"]))
        Problem["Stats"]["Std"].append(np.std(Population["f"]))

        # if Problem["Verbose"]:
        #     # Search illustration: plots the population if number of variables is 2
        #     if Problem["Verbose"] == 2:
        #         plt.pause(0.2)
        #         plt.title("Population at generation: " + str(Problem["Stats"]["GenCounter"]))
        #         p.set_xdata(Population["x"][:, 0])
        #         p.set_ydata(Population["x"][:, 1])
        #         p.set_color('b')
        #         plt.draw()

    toc()

    # Print if it was stopped due to the maximum of iterations or objective function evaluations
    if Problem["Stats"]["GenCounter"] >= MaxGenerations or Problem["Stats"]["ObjFunCounter"] >= MaxEvals:
        print("Maximum number of iterations or objective function evaluations reached")

    # Return best chromosome and objective function value
    BestChrom = [Population["x"][0]]
    BestChromObj = Population["f"][0]
    RunData = Problem["Stats"]

    # if Problem["Verbose"]:
    #     # Display some statistics
    #     if Problem["Verbose"] == 2:
    #         # Search illustration: plots the population if number of variables is 2
    #         title = "Population at generation: " + str(Problem["Stats"]["GenCounter"])
    #         p.set_xdata(Population["x"][0][0])
    #         p.set_ydata(Population["x"][0][1])
    #         p.set_color("k")
    #         plt.title(title)
    #         plt.draw()
    #         plt.pause(0.01)
    #     print(Problem["Stats"])

    return BestChrom, BestChromObj, RunData

def sphere_function(x):
    return np.sum(x ** 2)

Problem={'Variables': 2, 'ObjFunction':sphere_function , 'LB':[-5,-5], 'UB':[5,5]}
print(rGA(Problem))