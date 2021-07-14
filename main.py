from collections import deque
from copy import deepcopy
from datetime import datetime
from torch.optim import Adam
import gym
import imageio
import matplotlib.pyplot as plot
import numpy
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ReplayBuffer:
    def __init__(self, device, maximum_size=1_000_000):
        self.device = device
        self.buffer = deque(maxlen=maximum_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, numpy.array([reward]), next_state, numpy.array([done]))
        self.buffer.append(experience)

    def sample(self, batch_size=256):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return {"state": torch.FloatTensor(state_batch).to(self.device),
                "action": torch.FloatTensor(action_batch).to(self.device),
                "reward": torch.FloatTensor(reward_batch).to(self.device),
                "next_state": torch.FloatTensor(next_state_batch).to(self.device),
                "done": torch.FloatTensor(done_batch).to(self.device)}

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck noise
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, maximum_sigma=0.3, minimum_sigma=0.3, decay_period=100_000):
        self.mu = mu
        self.theta = theta
        self.sigma = maximum_sigma
        self.maximum_sigma = maximum_sigma
        self.minimum_sigma = minimum_sigma
        self.decay_period = decay_period
        self.dimensions = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = numpy.array([])
        self.reset()

    def reset(self):
        self.state = numpy.ones(self.dimensions) * self.mu

    def evolve_state(self):
        return self.state + self.theta * (self.mu - self.state) + self.sigma * numpy.random.randn(self.dimensions)

    def get_action(self, action, step=0):
        ou_state = self.evolve_state()
        self.sigma = self.maximum_sigma - (self.maximum_sigma - self.minimum_sigma) * min(1.0, step / self.decay_period)
        return numpy.clip(action + ou_state, self.low, self.high)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, first_hidden_size, second_hidden_size):
        super().__init__()
        self.first_hidden_layer = nn.Linear(state_size, first_hidden_size)
        self.second_hidden_layer = nn.Linear(first_hidden_size, second_hidden_size)
        self.output_layer = nn.Linear(second_hidden_size, action_size)

    def forward(self, state):
        first_hidden_layer_output = functional.relu(self.first_hidden_layer(state))
        second_hidden_layer_output = functional.relu(self.second_hidden_layer(first_hidden_layer_output))
        return torch.tanh(self.output_layer(second_hidden_layer_output))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, first_hidden_size, second_hidden_size):
        super().__init__()
        self.first_hidden_layer = nn.Linear(state_size + action_size, first_hidden_size)
        self.second_hidden_layer = nn.Linear(first_hidden_size, second_hidden_size)
        self.output_layer = nn.Linear(second_hidden_size, 1)

    def forward(self, state, action):
        first_hidden_layer_output = functional.relu(self.first_hidden_layer(torch.cat([state, action], dim=-1)))
        second_hidden_layer_output = functional.relu(self.second_hidden_layer(first_hidden_layer_output))
        return self.output_layer(second_hidden_layer_output)


class DDPG:
    def __init__(self, device, environment):
        self.device = device
        self.environment = environment
        self.seed = 0
        self.replay_buffer_maximum_size = 1_000_000
        self.replay_buffer_minimum_size = 1_000
        self.exploitation = 0.01
        self.exploration_episodes = 0
        self.batch_size = 100
        self.actor_first_hidden_size = 256
        self.actor_second_hidden_size = 256
        self.critic_first_hidden_size = 256
        self.critic_second_hidden_size = 256
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.99
        self.polyak = 0.995

        self.environment.action_space.seed(self.seed)
        torch.manual_seed(self.seed)
        numpy.random.seed(self.seed)

        self.replay_buffer = ReplayBuffer(self.device, self.replay_buffer_maximum_size)
        self.noise = OUNoise(self.environment.action_space)

        self.state_size = self.environment.observation_space.shape[0]
        self.action_size = self.environment.action_space.shape[0]

        self.actor = Actor(self.state_size, self.action_size, self.actor_first_hidden_size,
                           self.actor_second_hidden_size).to(device)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.critic = Critic(self.state_size, self.action_size, self.critic_first_hidden_size,
                             self.critic_second_hidden_size).to(device)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def save_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_critic_loss(self, samples):
        states, actions, rewards, next_states, dones = \
            samples["state"], samples["action"], samples["reward"], samples["next_state"], samples["done"]
        value = self.critic(states, actions)
        with torch.no_grad():
            target_action = self.target_actor(next_states)
            target_value = rewards + self.discount_factor * (1 - dones) * self.target_critic(next_states, target_action)
        return functional.mse_loss(value, target_value)

    def compute_actor_loss(self, states):
        return -self.critic(states, self.actor(states)).mean()

    def minimize_critic_loss(self, samples):
        self.critic_optimizer.zero_grad()
        self.compute_critic_loss(samples).backward()
        self.critic_optimizer.step()

    def minimize_actor_loss(self, samples):
        self.actor_optimizer.zero_grad()
        self.compute_actor_loss(samples["state"]).backward()
        self.actor_optimizer.step()

    def update_targets(self):
        with torch.no_grad():
            for critic_parameter, target_critic_parameter in zip(self.critic.parameters(),
                                                                 self.target_critic.parameters()):
                target_critic_parameter.mul_(self.polyak)
                target_critic_parameter.add_((1 - self.polyak) * critic_parameter)
            for actor_parameter, target_actor_parameter in zip(self.actor.parameters(),
                                                               self.target_actor.parameters()):
                target_actor_parameter.mul_(self.polyak)
                target_actor_parameter.add_((1 - self.polyak) * actor_parameter)

    def update(self, samples):
        self.minimize_critic_loss(samples)
        self.minimize_actor_loss(samples)
        self.update_targets()

    def train(self, epochs=1):
        if len(self.replay_buffer) < self.replay_buffer_minimum_size:
            return
        for epoch in range(epochs):
            self.update(self.replay_buffer.sample(self.batch_size))

    def get_train_action(self, state, episode, step):
        if episode <= self.exploration_episodes:
            exploration = max((self.exploitation - 1) / self.exploration_episodes * episode + 1, self.exploitation)
            if random.random() < exploration:
                return self.environment.action_space.sample()
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).to(self.device)).cpu().numpy()
        return self.noise.get_action(action, step)

    def get_test_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).to(self.device)).cpu().numpy()
        return action

    def save_models(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.target_actor.state_dict(), os.path.join(path, "target actor.pth"))

        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.target_critic.state_dict(), os.path.join(path, "target critic.pth"))

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"),
                                              map_location=lambda storage, loc: storage))
        self.target_actor.load_state_dict(torch.load(os.path.join(path, "target actor.pth"),
                                                     map_location=lambda storage, loc: storage))

        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"),
                                               map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(torch.load(os.path.join(path, "target critic.pth"),
                                                      map_location=lambda storage, loc: storage))


class TD3:
    def __init__(self, device, environment):
        self.device = device
        self.environment = environment
        self.seed = 0
        self.replay_buffer_maximum_size = 1_000_000
        self.replay_buffer_minimum_size = 1_000
        self.exploitation = 0.01
        self.exploration_episodes = 0
        self.batch_size = 100
        self.actor_first_hidden_size = 256
        self.actor_second_hidden_size = 256
        self.critic_first_hidden_size = 256
        self.critic_second_hidden_size = 256
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.99
        self.polyak = 0.995
        self.policy_delay = 2

        self.environment.action_space.seed(self.seed)
        torch.manual_seed(self.seed)
        numpy.random.seed(self.seed)

        self.replay_buffer = ReplayBuffer(self.device, self.replay_buffer_maximum_size)
        self.noise = OUNoise(self.environment.action_space)

        self.state_size = self.environment.observation_space.shape[0]
        self.action_size = self.environment.action_space.shape[0]

        self.actor = Actor(self.state_size, self.action_size, self.actor_first_hidden_size,
                           self.actor_second_hidden_size).to(device)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.first_critic = Critic(self.state_size, self.action_size, self.critic_first_hidden_size,
                                   self.critic_second_hidden_size).to(device)
        self.first_target_critic = deepcopy(self.first_critic)
        self.first_critic_optimizer = Adam(self.first_critic.parameters(), lr=self.critic_learning_rate)

        self.second_critic = Critic(self.state_size, self.action_size, self.critic_first_hidden_size,
                                    self.critic_second_hidden_size).to(device)
        self.second_target_critic = deepcopy(self.second_critic)
        self.second_critic_optimizer = Adam(self.second_critic.parameters(), lr=self.critic_learning_rate)

    def save_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_critic_loss(self, samples, step):
        states, actions, rewards, next_states, dones = \
            samples["state"], samples["action"], samples["reward"], samples["next_state"], samples["done"]
        first_value = self.first_critic(states, actions)
        second_value = self.second_critic(states, actions)
        with torch.no_grad():
            target_action = torch.FloatTensor(self.noise.get_action(self.target_actor(next_states).cpu().numpy(),
                                                                    step)).to(self.device)
            first_target_value = self.first_target_critic(next_states, target_action)
            second_target_value = self.second_target_critic(next_states, target_action)
            target_value = rewards + self.discount_factor * (1 - dones) * torch.min(first_target_value,
                                                                                    second_target_value)
        return functional.mse_loss(first_value, target_value), functional.mse_loss(second_value, target_value)

    def compute_actor_loss(self, states):
        return -self.first_critic(states, self.actor(states)).mean()

    def minimize_critic_loss(self, samples, step):
        first_critic_loss, second_critic_loss = self.compute_critic_loss(samples, step)
        self.first_critic_optimizer.zero_grad()
        first_critic_loss.backward()
        self.first_critic_optimizer.step()
        self.second_critic_optimizer.zero_grad()
        second_critic_loss.backward()
        self.second_critic_optimizer.step()

    def minimize_actor_loss(self, samples):
        self.actor_optimizer.zero_grad()
        self.compute_actor_loss(samples["state"]).backward()
        self.actor_optimizer.step()

    def update_targets(self):
        with torch.no_grad():
            for first_critic_parameter, first_target_critic_parameter in zip(self.first_critic.parameters(),
                                                                             self.first_target_critic.parameters()):
                first_target_critic_parameter.mul_(self.polyak)
                first_target_critic_parameter.add_((1 - self.polyak) * first_critic_parameter)
            for second_critic_parameter, second_target_critic_parameter in zip(self.second_critic.parameters(),
                                                                               self.second_target_critic.parameters()):
                second_target_critic_parameter.mul_(self.polyak)
                second_target_critic_parameter.add_((1 - self.polyak) * second_critic_parameter)
            for actor_parameter, target_actor_parameter in zip(self.actor.parameters(),
                                                               self.target_actor.parameters()):
                target_actor_parameter.mul_(self.polyak)
                target_actor_parameter.add_((1 - self.polyak) * actor_parameter)

    def update(self, samples, step):
        self.minimize_critic_loss(samples, step)
        if step % self.policy_delay == 0:
            self.minimize_actor_loss(samples)
            self.update_targets()

    def train(self, step, epochs=1):
        if len(self.replay_buffer) < self.replay_buffer_minimum_size:
            return
        for epoch in range(epochs):
            self.update(self.replay_buffer.sample(self.batch_size), step)

    def get_train_action(self, state, episode, step):
        if episode <= self.exploration_episodes:
            exploration = max((self.exploitation - 1) / self.exploration_episodes * episode + 1, self.exploitation)
            if random.random() < exploration:
                return self.environment.action_space.sample()
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).to(self.device)).cpu().numpy()
        return self.noise.get_action(action, step)

    def get_test_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).to(self.device)).cpu().numpy()
        return action

    def save_models(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.target_actor.state_dict(), os.path.join(path, "target actor.pth"))

        torch.save(self.first_critic.state_dict(), os.path.join(path, "first critic.pth"))
        torch.save(self.first_target_critic.state_dict(), os.path.join(path, "first target critic.pth"))

        torch.save(self.second_critic.state_dict(), os.path.join(path, "second critic.pth"))
        torch.save(self.second_target_critic.state_dict(), os.path.join(path, "second target critic.pth"))

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"),
                                              map_location=lambda storage, loc: storage))
        self.target_actor.load_state_dict(torch.load(os.path.join(path, "target actor.pth"),
                                                     map_location=lambda storage, loc: storage))

        self.first_critic.load_state_dict(torch.load(os.path.join(path, "first critic.pth"),
                                                     map_location=lambda storage, loc: storage))
        self.first_target_critic.load_state_dict(torch.load(os.path.join(path, "first target critic.pth"),
                                                            map_location=lambda storage, loc: storage))

        self.second_critic.load_state_dict(torch.load(os.path.join(path, "second critic.pth"),
                                                      map_location=lambda storage, loc: storage))
        self.second_target_critic.load_state_dict(torch.load(os.path.join(path, "second target critic.pth"),
                                                             map_location=lambda storage, loc: storage))


def train(environment, agent, path, mode="easy", algorithm="DDPG",
          episodes=1_000, maximum_score=300, minimum_score_percentage=0.95, threshold=200):
    os.mkdir(os.path.join(path, "graphs"))
    os.mkdir(os.path.join(path, "models"))
    os.mkdir(os.path.join(path, "GIFs"))
    os.mkdir(os.path.join(path, "GIFs", "train"))
    information = {"scores": [], "steps": []}
    tested_score = 0
    for episode in range(1, episodes + 1):
        images = []
        steps = 0
        state = environment.reset()
        score = 0
        done = False
        start_time = time.time()
        while not done:
            steps += 1
            action = agent.get_train_action(state, episode, steps)
            next_state, reward, done, _ = environment.step(action)
            agent.save_experience(state, action, reward, next_state, done)
            if algorithm == "DDPG":
                agent.train((episode - 1) // threshold + 1)
            else:
                agent.train(steps, (episode - 1) // threshold + 1)
            state = next_state
            score += reward
            images += [environment.render(mode="rgb_array")]
        end_time = time.time()
        information["scores"] += [score]
        information["steps"] += [steps]
        if score >= maximum_score:
            print("Episode: %4d, score: %8.3f, steps: %4d, elapsed time: %7.3f (SOLVED)" % (episode, score, steps,
                                                                                            end_time - start_time))
        else:
            print("Episode: %4d, score: %8.3f, steps: %4d, elapsed time: %7.3f" % (episode, score, steps,
                                                                                   end_time - start_time))
        if score >= minimum_score_percentage * maximum_score:
            score = str("%7.3f" % score).replace('.', '-')
            if not os.path.exists(os.path.join(path, "models", score)):
                os.mkdir(os.path.join(path, "models", score))
                agent.save_models(os.path.join(path, "models", score))
                imageio.mimsave(os.path.join(path, "GIFs", "train", score + ".gif"),
                                [numpy.array(image) for image in images], fps=30)
            tested_score = score
    best = "Best score: %f, related episode: %d, related steps: %d" % (max(information["scores"]),
                                                                       numpy.argmax(information["scores"]) + 1,
                                                                       information["steps"]
                                                                       [numpy.argmax(information["scores"]).item()])
    print(best)
    if mode == "easy":
        mode = "Bipedal Walker"
    else:
        mode = "Bipedal Walker Hardcore"
    if algorithm == "DDPG":
        algorithm = "Deep Deterministic Policy Gradient"
    else:
        algorithm = "Twin Delayed Deep Deterministic Policy Gradient"
    plot.plot(range(1, episodes + 1), information["scores"], "purple")
    plot.xlabel("Episode")
    plot.ylabel("Score")
    plot.title("%s\n(%s)" % (mode, algorithm))
    plot.savefig(os.path.join(path, "graphs", "scores.png"), dpi=300)
    plot.show()
    plot.plot(range(1, episodes + 1), information["steps"], "green")
    plot.xlabel("Episode")
    plot.ylabel("Steps")
    plot.title("%s\n(%s)" % (mode, algorithm))
    plot.savefig(os.path.join(path, "graphs", "steps.png"), dpi=300)
    plot.show()
    return tested_score


def test(environment, agent, path, tested_score, episodes=10, maximum_score=300, minimum_score_percentage=0.95):
    if tested_score == 0:
        return
    if not os.path.exists(os.path.join(path, "GIFs", "test")):
        os.mkdir(os.path.join(path, "GIFs", "test"))
    if not os.path.exists(os.path.join(path, "GIFs", "test", tested_score)):
        os.mkdir(os.path.join(path, "GIFs", "test", tested_score))
    agent.load_models(os.path.join(path, "models", tested_score))
    for episode in range(1, episodes + 1):
        images = []
        steps = 0
        state = environment.reset()
        score = 0
        done = False
        start_time = time.time()
        while not done:
            steps += 1
            action = agent.get_test_action(state)
            state, reward, done, _ = environment.step(action)
            score += reward
            images += [environment.render(mode="rgb_array")]
        end_time = time.time()
        if score >= maximum_score:
            print("Episode: %4d, score: %8.3f, steps: %4d, elapsed time: %7.3f (SOLVED)" % (episode, score, steps,
                                                                                            end_time - start_time))
        else:
            print("Episode: %4d, score: %8.3f, steps: %4d, elapsed time: %7.3f" % (episode, score, steps,
                                                                                   end_time - start_time))
        if score >= minimum_score_percentage * maximum_score:
            score = str("%7.3f" % score).replace('.', '-')
            imageio.mimsave(os.path.join(path, "GIFs", "test", tested_score, score + ".gif"),
                            [numpy.array(image) for image in images], fps=30)


def run(device, mode="easy", algorithm="DDPG", minimum_score_percentage=0.95, train_agent=True, test_agent=True):
    if mode == "easy":
        environment = gym.make("BipedalWalker-v3")
        print("Environment: Bipedal Walker")
    else:
        environment = gym.make("BipedalWalkerHardcore-v3")
        print("Environment: Bipedal Walker Hardcore")
    if algorithm == "DDPG":
        agent = DDPG(device, environment)
        print("Algorithm: Deep Deterministic Policy Gradient")
    else:
        agent = TD3(device, environment)
        print("Algorithm: Twin Delayed Deep Deterministic Policy Gradient")
    if train_agent:
        print("Phase: training")
        path = os.path.join(os.getcwd(), mode, algorithm, datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        os.mkdir(path)
        tested_score = train(environment, agent, path, mode, algorithm,
                             episodes=1_000, maximum_score=300, minimum_score_percentage=minimum_score_percentage,
                             threshold=100)
        if test_agent:
            print("Phase: testing")
            test(environment, agent, path, tested_score, episodes=10, maximum_score=300,
                 minimum_score_percentage=minimum_score_percentage)
    elif test_agent:
        print("Phase: testing")
        path = input("Insert path: ")
        tested_score = input("Insert tested score: ")
        test(environment, agent, path, tested_score, episodes=10, maximum_score=300,
             minimum_score_percentage=minimum_score_percentage)
    environment.close()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    run(device, mode="easy", algorithm="DDPG", minimum_score_percentage=0.9, train_agent=True, test_agent=True)
    run(device, mode="easy", algorithm="TD3", minimum_score_percentage=1, train_agent=True, test_agent=True)
    run(device, mode="hard", algorithm="DDPG", minimum_score_percentage=0, train_agent=True, test_agent=True)
    run(device, mode="hard", algorithm="TD3", minimum_score_percentage=0, train_agent=True, test_agent=True)


if __name__ == '__main__':
    main()
