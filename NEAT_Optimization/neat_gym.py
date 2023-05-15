import argparse
import os

import gym
import numpy as np

import neat
import visualize

game_name = 'SpaceInvaders-ram-v0'

max_steps = 10000
episodes = 5
generations = 100
numCores = 4

def evaluate_fitness(net, env, episodes=1, steps=5000, render=False):
    # Specify seed below if the same seed is required for every game.
    # env.seed(0)  Have same seed for all game
    fitnesses = []
    for episode in range(episodes):
        inputs = game_env.reset()  # Receive observation from OpenAI as Inputs
        total_reward = 0.0

        for j in range(steps):
            outputs = net.activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, info = env.step(action)
            if done:
                break
            total_reward += reward

        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    return fitness


def evaluate_genome(g, conf):
    """ Send a genome & config file to the neat implementation and receive its phenotype (a FeedForwardNetwork). """
    net = neat.nn.FeedForwardNetwork.create(g, conf)
    return evaluate_fitness(net, game_env, episodes, max_steps, render=False)


def run_neat(env):
    def eval_genomes(genomes, conf):
        for g in genomes:
            fitness = evaluate_genome(g, conf)
            g.fitness = fitness

    # Locate config file & Load
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'openAI_NEAT_config')
    conf = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create population from config file
    pop = neat.population.Population(conf)

    # Create Statistics reporter
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Create automatic checkpointer to create checkpoints
    pop.add_reporter(neat.Checkpointer(10, 900))

    parallel_evaluator = neat.parallel.ParallelEvaluator(numCores, evaluate_genome)
    pop.run(parallel_evaluator.evaluate, generations)

    stats.save()

    # Show output of the current most fit genome against training data.
    best_genome = pop.best_genome

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(best_genome))
    # Show output of the most fit genome against training data.
    print('\nOutput:')

    visualize.draw_net(conf, best_genome, True, filename="winner_net")
    visualize.plot_stats(stats, ylog=False, view=False, filename="stats.png")
    visualize.plot_species(stats, view=True)

    # Give option to run fittest individual again
    best_genome_net = neat.nn.FeedForwardNetwork.create(best_genome, conf)
    for i in range(100):
        evaluate_fitness(best_genome_net, env, 1, max_steps, render=True)

    env.close()


game_env = gym.make(game_name)
print ("Input Nodes: %s" % str(len(game_env.observation_space.high)))
print ("Output Nodes: %s" % str(game_env.action_space.n))
print("Action space: {0!r}".format(game_env.action_space))
print("Observation space: {0!r}".format(game_env.observation_space))
run_neat(game_env)
