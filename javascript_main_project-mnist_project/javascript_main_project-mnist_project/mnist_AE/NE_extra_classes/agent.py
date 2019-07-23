from NE_extra_classes.brain import Brain
class Agent:

    def __init__(self, fs_template, vs_template=None, ms_template=None):

        self.visual_system = vs_template
        self.functional_system = fs_template
        self.memory_system = ms_template

        self.fitness = None
        self.fitness_over_time = []


