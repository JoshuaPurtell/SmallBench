from abc import ABC, abstractmethod


class Agent:
    @abstractmethod
    async def act(self):
        pass

    @abstractmethod
    def add_observation(self, observation):
        pass
