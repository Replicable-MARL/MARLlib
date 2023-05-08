from .voltage_barrier_registry import Voltage_Barrier



class VoltageBarrier(object):
    def __init__(self, name):
        self.name = name
        self.voltage_barrier = Voltage_Barrier[name]

    def step(self, vs):
        return self.voltage_barrier(vs)
